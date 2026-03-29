"""
Universal Data Sanitization API — FastAPI entry point.

Provides anomaly detection over arbitrary tabular / embedding data
using PyTorch (Autoencoder, Deep SVDD) and scikit-learn (Isolation Forest).

Thresholding: Z-score based (mean + 2·σ).  If data is uniform, 0 flags.
Ensemble:     ≥ 2 / 3 model votes → "poisoned".
"""

from __future__ import annotations

import io
import json
import random
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from extractor import UniversalExtractor
from models import DynamicAutoencoder, DynamicDeepSVDD

# ---------------------------------------------------------------------------
# Strict Determinism
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Universal Data Sanitization API",
    description="Detect and sanitize anomalous data using ensemble anomaly detection.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared extractor instance (ResNet lazy-loaded on first image ZIP)
extractor = UniversalExtractor()

# Training hyper-parameters
EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAD_THRESHOLD = 4.5

# Model display names (used in the response payload)
MODEL_AE = "Autoencoder"
MODEL_SVDD = "Deep SVDD"
MODEL_ISO = "Isolation Forest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_mad_scores_and_flags(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Median Absolute Deviation (MAD) of scores and flag robust outliers."""
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    
    if mad < 1e-9:
        return scores, np.zeros(len(scores), dtype=np.int32)
        
    modified_z = 0.6745 * (scores - median) / mad
    flags = (modified_z > MAD_THRESHOLD).astype(np.int32)
    return scores, flags


def _train_autoencoder(
    tensor_data: torch.Tensor, input_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a DynamicAutoencoder with Early Stopping; return (raw_scores, mad_flags, reconstructions)."""
    model = DynamicAutoencoder(input_dim)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(tensor_data), batch_size=BATCH_SIZE, shuffle=True
    )

    best_loss = float("inf")
    patience_counter = 0

    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(loader)
        if best_loss - epoch_loss > 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 2:
                break

    model.eval()
    with torch.no_grad():
        reconstructions = model(tensor_data).numpy()
        errors = model.reconstruction_error(tensor_data).numpy()

    scores, flags = _calculate_mad_scores_and_flags(errors)
    return scores, flags, reconstructions


def _train_deep_svdd(
    tensor_data: torch.Tensor, input_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a DynamicDeepSVDD with Early Stopping; return (raw_scores, mad_flags)."""
    model = DynamicDeepSVDD(input_dim)

    # Initialize center c
    model.eval()
    with torch.no_grad():
        initial_out = model(tensor_data)
        model.center.copy_(initial_out.mean(dim=0))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loader = DataLoader(
        TensorDataset(tensor_data), batch_size=BATCH_SIZE, shuffle=True
    )

    best_loss = float("inf")
    patience_counter = 0

    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = torch.mean(torch.sum((output - model.center) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(loader)
        if best_loss - epoch_loss > 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 2:
                break

    model.eval()
    with torch.no_grad():
        distances = model.anomaly_score(tensor_data).numpy()

    return _calculate_mad_scores_and_flags(distances)


def _run_isolation_forest(feature_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit an IsolationForest and return (inverted_score_samples, binary_flags)."""
    # Strict dynamic contamination for small datasets
    contam = max(0.001, min(0.05, 10.0 / len(feature_vectors)))
    iso = IsolationForest(contamination=contam, random_state=42)
    preds = iso.fit_predict(feature_vectors)
    
    # sklearn: -1 → anomaly, 1 → normal
    flags = (preds == -1).astype(np.int32)
    # score_samples are negative; lower is more anomalous.
    # Invert so higher is more anomalous, aligning precisely with AE and SVDD.
    raw_scores = -iso.score_samples(feature_vectors)
    
    return raw_scores, flags


def _generate_xai_payload(
    ident: str, orig_f: np.ndarray, recon_f: np.ndarray, median_vector: np.ndarray
) -> dict:
    """Generate the 'Why' explanation using Latent Features or Tabular Feature Delta."""
    if extractor.last_images and ident in extractor.last_images:
        diff = np.abs(orig_f - median_vector)
        top_indices = np.argsort(diff)[::-1][:10]
        explanations = [{"feature_index": int(idx), "deviation_score": float(diff[idx])} for idx in top_indices]
        return {
            "type": "image",
            "top_contributors": explanations
        }
        
    elif extractor.last_columns:
        diff = np.abs(orig_f - recon_f)
        cols = extractor.last_columns
        error_map = {}
        # Aggregate expanded OneHot columns back to original names
        for idx, col in enumerate(cols):
            base_col = col.split("_")[0] if "_" in col else col
            error_map[base_col] = error_map.get(base_col, 0.0) + float(diff[idx])
            
        sorted_errors = sorted(error_map.items(), key=lambda x: x[1], reverse=True)
        top_3 = [{"name": k, "error": v} for k, v in sorted_errors[:3]]
        return {
            "type": "tabular",
            "top_contributors": top_3
        }
    return {}

def _build_flagged_items(
    identifiers: List[str],
    ae_flags: np.ndarray,
    svdd_flags: np.ndarray,
    iso_flags: np.ndarray,
    combined_norm: np.ndarray,
    feature_vectors: np.ndarray,
    reconstructions: np.ndarray,
    median_vector: np.ndarray,
) -> list[dict]:
    """
    Build the enriched flagged-items list.

    A sample is poisoned if its combined average score > 0.7 
    OR if >= 2 models flagged it individually.
    """
    model_names = [MODEL_AE, MODEL_SVDD, MODEL_ISO]
    items: list[dict] = []

    for i, ident in enumerate(identifiers):
        flags = [ae_flags[i], svdd_flags[i], iso_flags[i]]
        
        is_poisoned = bool(combined_norm[i] > 0.7) or (sum(flags) >= 2)
        if not is_poisoned:
            continue
            
        flagged_by = [name for name, f in zip(model_names, flags) if f]
        explanation = _generate_xai_payload(ident, feature_vectors[i], reconstructions[i], median_vector)
        
        items.append({
            "id": ident, 
            "flagged_by": flagged_by,
            "explanation": explanation
        })

    return items


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "healthy", "version": app.version}


@app.post("/scan-dataset")
async def scan_dataset(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV or ZIP, run an ensemble of three anomaly detectors,
    and return the flagged (poisoned) samples with per-model attribution.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1073741824:
        raise HTTPException(status_code=413, detail="Dataset exceeds 1GB limit.")

    try:
        # 1. Extract features
        feature_vectors, identifiers = await extractor.process_upload(file)
        n_samples, input_dim = feature_vectors.shape

        # 2. Prepare PyTorch tensor
        tensor_data = torch.tensor(feature_vectors, dtype=torch.float32)

        # 3. Run all three detectors
        ae_scores, ae_flags, ae_recons = _train_autoencoder(tensor_data, input_dim)
        svdd_scores, svdd_flags = _train_deep_svdd(tensor_data, input_dim)
        iso_scores, iso_flags = _run_isolation_forest(feature_vectors)

        # 4. Model Normalization
        scaler = MinMaxScaler()
        ae_norm = scaler.fit_transform(ae_scores.reshape(-1, 1)).flatten()
        svdd_norm = scaler.fit_transform(svdd_scores.reshape(-1, 1)).flatten()
        iso_norm = scaler.fit_transform(iso_scores.reshape(-1, 1)).flatten()
        
        # 5. Combined Average Normalized Score
        combined_norm = (ae_norm + svdd_norm + iso_norm) / 3.0

        # Calculate median vector for Image XAI
        median_vector = np.median(feature_vectors, axis=0)

        # 6. Build enriched results
        flagged_items = _build_flagged_items(
            identifiers, ae_flags, svdd_flags, iso_flags, combined_norm, feature_vectors, ae_recons, median_vector
        )
        poisoned_count = len(flagged_items)
        anomaly_pct = round((poisoned_count / n_samples) * 100, 2)

        return {
            "total_samples": n_samples,
            "poisoned_samples": poisoned_count,
            "anomaly_percentage": anomaly_pct,
            "model_breakdown": {
                MODEL_AE: int(ae_flags.sum()),
                MODEL_SVDD: int(svdd_flags.sum()),
                MODEL_ISO: int(iso_flags.sum()),
            },
            "flagged_items": flagged_items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/download-sanitized")
async def download_sanitized(
    request: Request,
    file: UploadFile = File(...),
    flagged_items: str = Form(...),
):
    """
    Accept the original file and a JSON list of flagged IDs.
    Return a cleaned version with the flagged items removed.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1073741824:
        raise HTTPException(status_code=413, detail="Dataset exceeds 1GB limit.")

    try:
        to_remove: list[str] = json.loads(flagged_items)
    except (json.JSONDecodeError, TypeError):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="flagged_items must be valid JSON.")

    remove_set = set(to_remove)
    filename = (file.filename or "").lower()
    raw = await file.read()

    # ── CSV path ──────────────────────────────────────────────────────
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        # Row indices were stringified in the scan response
        mask = ~df.index.astype(str).isin(remove_set)
        clean_df = df.loc[mask]

        buf = io.StringIO()
        clean_df.to_csv(buf, index=False)
        buf.seek(0)

        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=sanitized_data.csv"
            },
        )

    # ── ZIP path ──────────────────────────────────────────────────────
    if filename.endswith(".zip"):
        src = zipfile.ZipFile(io.BytesIO(raw))
        out_buf = io.BytesIO()

        with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as dst:
            for entry in src.namelist():
                # Skip directories, macOS junk, and flagged files
                if entry.endswith("/") or entry.startswith("__MACOSX"):
                    continue
                basename = entry.split("/")[-1]
                if basename in remove_set:
                    continue
                dst.writestr(entry, src.read(entry))

        out_buf.seek(0)

        return StreamingResponse(
            out_buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=sanitized_data.zip"
            },
        )

    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail="Unsupported file type.")
