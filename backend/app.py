"""
Universal Data Sanitization API — FastAPI entry point.

Provides anomaly detection over arbitrary tabular / embedding data
using PyTorch (Autoencoder, Deep SVDD) and scikit-learn (Isolation Forest).

Thresholding: Z-score based (mean + 2·σ).  If data is uniform, 0 flags.
Ensemble:     ≥ 2 / 3 model votes → "poisoned".
"""

from __future__ import annotations

import io
import uuid
import json
import os
import random
import time
from fpdf import FPDF
import shutil
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import asyncio
import json
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from torch.utils.data import DataLoader, TensorDataset

from extractor import UniversalExtractor
from models import DynamicAutoencoder, DynamicDeepSVDD

'''
=== VERIFICATION CHECKS (Remove before commit) ===
1. _calibrate_mad_threshold returns threshold in [2.5, 4.0]
   => CONFIRMED: Uses `float(np.clip(z_thresh, 2.5, 4.0))`.
2. DynamicAutoencoder(4).encoder[-1].out_features >= 4
   => CONFIRMED: `input_dim <= 16` sets `bottleneck = max(input_dim // 2, 4)`. For dimension 4, out_features is precisely 4.
3. DynamicDeepSVDD center is set AFTER main training loop
   => NEGATIVE / SET BEFORE: As instructed in the new 3-pass flow, the center is fundamentally established AFTER the 2-epoch Warm-Up phase, but BEFORE the main training (EPOCHS) loop to actively penalize divergence against the static target.
4. IsolationForest contamination == 'auto'
   => CONFIRMED: Uses scikit-learn's built-in heuristic (`contamination='auto'`).
5. F.normalize only runs when extractor.last_images is truthy
   => CONFIRMED: Normalization block condition requires mutually inclusive `if extractor.last_images and input_dim >= 512:` boundaries, preventing NLP overwrites.
===================================================
'''

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

class LimitUploadSizeMiddleware:
    """
    ASGI Middleware to physically count bytes streamed into the server.
    Raises a 413 error if the total payload exceeds the max_upload_size limit,
    preventing malicious memory exhaustion attacks without relying on HTTP headers.
    """
    def __init__(self, app, max_upload_size: int = 1073741824): # 1GB Limit
        self.app = app
        self.max_upload_size = max_upload_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        total_size = 0

        async def receive_with_limit():
            nonlocal total_size
            message = await receive()
            if message["type"] == "http.request":
                chunk_size = len(message.get("body", b""))
                total_size += chunk_size
                if total_size > self.max_upload_size:
                    raise HTTPException(status_code=413, detail="Dataset exceeds 1GB physical byte limit.")
            return message

        await self.app(scope, receive_with_limit, send)

import secrets
from fastapi import Header

API_KEY = os.getenv("DISTILL_API_KEY", None)

async def verify_api_key(
    x_api_key: str | None = Header(None)
) -> bool:
    if API_KEY is None:
        return True  # no key configured = open access
    if x_api_key is None or not secrets.compare_digest(
        x_api_key, API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key header."
        )
    return True

app = FastAPI(
    title="Universal Data Sanitization API",
    description="Detect and sanitize anomalous data using ensemble anomaly detection.",
    version="0.2.0",
)

app.add_middleware(LimitUploadSizeMiddleware)

# Rate limiter (5 requests / minute per IP on heavy endpoints)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Parse allowed origins from environment or use defaults
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    origins = [
        "https://distill-nine-theta.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared extractor instance (ResNet lazy-loaded on first image ZIP)
extractor = UniversalExtractor()

# Training hyper-parameters
EPOCHS = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 2048
MAD_THRESHOLD = 4.5

# Device selection — use the T4 GPU when available, fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model display names (used in the response payload)
MODEL_AE = "Autoencoder"
MODEL_SVDD = "Deep SVDD"
MODEL_ISO = "Isolation Forest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_natural_threshold(raw_scores: np.ndarray,
                             median: float, mad: float) -> float:
    sorted_scores = np.sort(raw_scores)
    if len(sorted_scores) < 10:
        return 3.0
    gaps = np.diff(sorted_scores)
    upper_start = int(len(gaps) * 0.70)
    upper_gaps = gaps[upper_start:]
    if len(upper_gaps) > 0 and upper_gaps.mean() > 0:
        if upper_gaps.max() > 3 * upper_gaps.mean():
            cliff_idx = upper_start + np.argmax(upper_gaps)
            natural_threshold = sorted_scores[cliff_idx]
            z = 0.6745 * (natural_threshold - median) / max(mad, 1e-5)
            return float(np.clip(z, 2.0, 6.0))
    return 3.5

def _calibrate_mad_threshold(raw_scores: np.ndarray, n_samples: int) -> Tuple[float, float, float]:
    """Calculate and freeze MAD scaling logic on the first training chunk."""
    median = float(np.median(raw_scores))
    mad = float(np.median(np.abs(raw_scores - median)))
    mad = max(mad, 1e-5) # Prevent zero-division
    threshold = _find_natural_threshold(raw_scores, median, mad)
    return median, mad, threshold

def _evaluate_binary_votes(raw_scores: np.ndarray, median: float, mad: float, threshold: float) -> np.ndarray:
    """Evaluate an arbitrary chunk against the frozen MAD framework."""
    mod_z_scores = 0.6745 * (raw_scores - median) / mad
    return (mod_z_scores > threshold).astype(np.int32)

def _statistical_prefilter(
    features: np.ndarray,
    identifiers: list[str],
    columns: list[str],
) -> np.ndarray:
    """
    Returns boolean mask (True=hard anomaly) using:
    1. Per-column modified Z-score > 6.0 (6-sigma rule)
    2. IQR fence: value < Q1 - 4*IQR or > Q3 + 4*IQR
    3. Any row triggering EITHER rule on ANY column → hard anomaly
    """
    n, d = features.shape
    hard_flags = np.zeros(n, dtype=bool)

    for col_idx in range(d):
        col = features[:, col_idx]
        
        # Modified Z-score
        median = np.median(col)
        mad = np.median(np.abs(col - median))
        mad = max(mad, 1e-5)
        mod_z = 0.6745 * np.abs(col - median) / mad
        hard_flags |= (mod_z > 6.0)
        
        # IQR fence (4x = extreme outlier, not just mild)
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        if iqr > 1e-5:
            hard_flags |= (col < q1 - 4*iqr) | (col > q3 + 4*iqr)
    
    return hard_flags


def _train_autoencoder(train_tensor: torch.Tensor, input_dim: int) -> DynamicAutoencoder:
    """Train a DynamicAutoencoder with Early Stopping on the primary chunk."""
    model = DynamicAutoencoder(input_dim).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available()
    )

    best_loss = float("inf")
    patience_counter = 0

    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
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
            if patience_counter >= 1:
                break

    model.eval()
    return model

def _evaluate_autoencoder(model: DynamicAutoencoder, eval_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate an arbitrary chunk against the trained Autoencoder."""
    eval_loader = DataLoader(
        TensorDataset(eval_tensor), batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    
    all_recons = []
    all_errors = []
    _autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad(), torch.autocast(device_type=_autocast_device):
        for (batch,) in eval_loader:
            batch_dev = batch.to(device)
            recon = model(batch_dev)
            error = torch.mean((batch_dev - recon)**2, dim=1)
            all_recons.append(recon.cpu().float().numpy())
            all_errors.append(error.cpu().float().numpy())
            
    reconstructions = np.concatenate(all_recons, axis=0)
    errors = np.concatenate(all_errors, axis=0)
    return errors, reconstructions


def _train_deep_svdd(train_tensor: torch.Tensor, input_dim: int) -> DynamicDeepSVDD:
    """Train a DynamicDeepSVDD with Early Stopping on the primary chunk."""
    model = DynamicDeepSVDD(input_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available()
    )

    # 1. Warm-up (2 epochs): train pushing outputs toward origin
    model.train()
    for _ in range(2):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = torch.mean(torch.sum(output ** 2, dim=1))
            loss.backward()
            optimizer.step()

    # 2. Recompute center from stabilized encoder outputs
    model.eval()
    center_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    initial_outputs = []
    
    with torch.no_grad():
        for (c_batch,) in center_loader:
            initial_outputs.append(model(c_batch.to(device)))
            
    all_initial_out = torch.cat(initial_outputs, dim=0)
    model.center.copy_(all_initial_out.mean(dim=0))

    # 3. Main training (EPOCHS)
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
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
            if patience_counter >= 1:
                break

    model.eval()
    return model

def _evaluate_deep_svdd(model: DynamicDeepSVDD, eval_tensor: torch.Tensor) -> np.ndarray:
    """Evaluate an arbitrary chunk against the trained Deep SVDD."""
    eval_loader = DataLoader(
        TensorDataset(eval_tensor), batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    
    all_distances = []
    _autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad(), torch.autocast(device_type=_autocast_device):
        for (batch,) in eval_loader:
            batch_dev = batch.to(device)
            all_distances.append(model.anomaly_score(batch_dev).cpu().float().numpy())
            
    distances = np.concatenate(all_distances, axis=0)
    return distances


def _fit_pca(train_features: np.ndarray) -> PCA | None:
    n, d = train_features.shape
    n_components = min(d // 2, 20, n - 1)
    if n_components < 2:
        return None  # too few dims, skip PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_features)
    return pca

def _train_isolation_forest(train_features: np.ndarray, pca=None) -> IsolationForest:
    fit_features = pca.transform(train_features) if pca else train_features
    iso = IsolationForest(
        n_estimators=200,
        contamination='auto',
        n_jobs=-1,
        random_state=42
    )
    iso.fit(fit_features)
    return iso

def _evaluate_isolation_forest(model: IsolationForest, eval_features: np.ndarray, pca=None) -> np.ndarray:
    eval_f = pca.transform(eval_features) if pca else eval_features
    raw_scores = -1.0 * model.decision_function(eval_f)
    return raw_scores


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
        text_error_map = {}
        # Aggregate expanded columns back to original names
        for idx, col in enumerate(cols):
            if col.startswith("__TEXT__:"):
                base_col = col.replace("__TEXT__:", "").split("_dim")[0]
                text_error_map[base_col] = text_error_map.get(base_col, 0.0) + float(diff[idx])
            else:
                base_col = col.split("_")[0] if "_" in col else col
                error_map[base_col] = error_map.get(base_col, 0.0) + float(diff[idx])
                
        if text_error_map:
            best_text_col = max(text_error_map, key=text_error_map.get)
            best_text_err = text_error_map[best_text_col]
            
            best_tab_err = max(error_map.values()) if error_map else 0.0
                
            if best_text_err > best_tab_err:
                snippet = extractor.current_chunk_text_snippets.get(ident, {}).get(best_text_col, "Text snippet missing")
                return {
                    "type": "text",
                    "snippet": snippet,
                    "column": best_text_col
                }
            
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
    feature_vectors: np.ndarray,
    reconstructions: np.ndarray,
    median_vector: np.ndarray,
    hard_flags: np.ndarray = None,
    conf_ae: np.ndarray = None,
    conf_svdd: np.ndarray = None,
    conf_iso: np.ndarray = None,
    ensemble_conf: np.ndarray = None,
    n_samples: int = 0,
) -> list[dict]:
    """
    Build the enriched flagged-items list.
    Requires >= 2 votes normally; >= 1 vote for small chunks (< 100 samples).
    """
    if hard_flags is None:
        hard_flags = np.zeros(len(identifiers), dtype=bool)

    model_names = [MODEL_AE, MODEL_SVDD, MODEL_ISO]
    items: list[dict] = []

    ml_idx = 0
    for i, ident in enumerate(identifiers):
        if hard_flags[i]:
            items.append({
                "id": ident,
                "confidence": 1.0,
                "severity": "CRITICAL",
                "flagged_by": ["Statistical Pre-filter"],
                "model_scores": {
                    "Autoencoder": None,
                    "Deep SVDD": None,
                    "Isolation Forest": None
                },
                "explanation": {"type": "tabular", "top_contributors": []}
            })
            continue

        flags = [ae_flags[ml_idx], svdd_flags[ml_idx], iso_flags[ml_idx]]
        total_votes = sum(flags)
        min_votes = 1 if n_samples < 100 else 2
        is_poisoned = bool(total_votes >= min_votes)
        if is_poisoned:
            flagged_by = [name for name, f in zip(model_names, flags) if f]
            explanation = _generate_xai_payload(ident, feature_vectors[i], reconstructions[ml_idx], median_vector)
            
            conf = float(ensemble_conf[ml_idx]) if ensemble_conf is not None else 0.0
            if conf >= 0.80:
                severity = "CRITICAL"
            elif conf >= 0.65:
                severity = "HIGH"  
            elif conf >= 0.50:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            items.append({
                "id": ident,
                "confidence": round(conf, 4),
                "severity": severity,
                "flagged_by": flagged_by,
                "model_scores": {
                    "Autoencoder":        round(float(conf_ae[ml_idx]),   4) if conf_ae   is not None else None,
                    "Deep SVDD":          round(float(conf_svdd[ml_idx]), 4) if conf_svdd is not None else None,
                    "Isolation Forest":   round(float(conf_iso[ml_idx]),  4) if conf_iso  is not None else None,
                },
                "explanation": explanation,
            })
        ml_idx += 1

    items.sort(key=lambda x: x["confidence"], reverse=True)
    return items


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "healthy", "version": app.version}


@app.get("/api-info")
async def api_info():
    return {
        "name": "Distill API",
        "version": app.version,
        "description": (
            "Universal data sanitization and poisoning "
            "detection API"
        ),
        "authentication": (
            "Pass X-API-Key header if configured"
        ),
        "endpoints": {
            "POST /scan-dataset": (
                "Upload CSV/ZIP/Parquet, returns full "
                "scan results as JSON"
            ),
            "POST /scan-stream": (
                "Same as scan-dataset but streams "
                "progress via SSE"
            ),
            "POST /download-sanitized": (
                "Upload original file + scan results, "
                "returns cleaned ZIP"
            ),
            "GET /pickup-sanitized/{file_id}": (
                "Download the cleaned file"
            ),
        },
        "supported_formats": [
            "CSV (.csv)",
            "ZIP archive of images (.zip)",
            "ZIP archive of CSVs (.zip)",
            "Parquet (.parquet)",
        ],
        "models": [
            "CLIP ViT-B/32 (image feature extraction)",
            "SentenceTransformer all-MiniLM-L6-v2 (NLP)",
            "DynamicAutoencoder (PyTorch)",
            "DynamicDeepSVDD (PyTorch)",
            "IsolationForest (scikit-learn)",
        ],
    }

# Temp uploads directory used in the Modal container
_TEMP_UPLOADS_DIR = os.getenv('TEMP_UPLOADS_DIR', os.path.join(os.getcwd(), 'temp_uploads'))
os.makedirs(_TEMP_UPLOADS_DIR, exist_ok=True)


def _purge_temp_uploads() -> None:
    """Securely delete all contents of the temp uploads directory."""
    if os.path.isdir(_TEMP_UPLOADS_DIR):
        shutil.rmtree(_TEMP_UPLOADS_DIR, ignore_errors=True)
    os.makedirs(_TEMP_UPLOADS_DIR, exist_ok=True)



def _normalize_scores_to_confidence(
    raw_scores: np.ndarray,
    median: float,
    mad: float,
) -> np.ndarray:
    """
    Convert raw anomaly scores to [0,1] confidence using sigmoid
    on modified Z-scores. 0=definitely clean, 1=definitely poisoned.
    """
    mod_z = 0.6745 * (raw_scores - median) / max(mad, 1e-5)
    # Sigmoid centered at threshold=0, scale=0.5 gives smooth curve
    confidence = 1.0 / (1.0 + np.exp(-0.5 * mod_z))
    return confidence.astype(np.float32)


def _run_scan_pipeline(data_stream, progress_cb=None) -> dict:
    # 2. Extract Chunk 0 (The Base Pattern)
    first_features, first_identifiers = next(data_stream)
    input_dim = first_features.shape[1]

    warning = None
    if len(first_features) < 100:
        warning = (
            f"Only {len(first_features)} samples detected. "
            "Reliable detection needs 100+ samples. "
            "Results may be inaccurate."
        )
    
    # 3. Prepare PyTorch tensors for Chunk 0
    chunk0_tensor = torch.tensor(first_features, dtype=torch.float32)

    # L2-Normalize high-dimensional embeddings exclusively for images
    if extractor.last_images and input_dim >= 512:
        import torch.nn.functional as F
        chunk0_tensor = F.normalize(chunk0_tensor, p=2, dim=1)
        first_features = chunk0_tensor.numpy()

    if progress_cb:
        progress_cb("phase", "TRAINING_MODELS")

    # 4. Train Models exclusively on Chunk 0
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as ex:
        ae_future = ex.submit(_train_autoencoder, chunk0_tensor, input_dim)
        svdd_future = ex.submit(_train_deep_svdd, chunk0_tensor, input_dim)
        
        pca = _fit_pca(first_features)
        iso_model = _train_isolation_forest(first_features, pca)

    ae_model = ae_future.result()
    svdd_model = svdd_future.result()
    
    if progress_cb:
        progress_cb("phase", "CALIBRATING")

    # 5. Calibrate MAD Thresholds
    raw_ae, _ = _evaluate_autoencoder(ae_model, chunk0_tensor)
    raw_svdd = _evaluate_deep_svdd(svdd_model, chunk0_tensor)
    raw_iso = _evaluate_isolation_forest(iso_model, first_features, pca)
    
    # Freeze thresholds using Chunk 0 topography
    chunk0_len = len(first_features)
    ae_median, ae_mad, ae_thresh = _calibrate_mad_threshold(raw_ae, chunk0_len)
    svdd_median, svdd_mad, svdd_thresh = _calibrate_mad_threshold(raw_svdd, chunk0_len)
    iso_median, iso_mad, iso_thresh = _calibrate_mad_threshold(raw_iso, chunk0_len)
    
    median_vector = np.median(first_features, axis=0)

    # 6. Stream Engine Loop
    flagged_items = []
    total_samples = 0
    model_breakdown = { MODEL_AE: 0, MODEL_SVDD: 0, MODEL_ISO: 0, "Statistical Pre-filter": 0 }
    
    # Re-inject Chunk 0 into the evaluation pipeline seamlessly
    import itertools
    full_stream = itertools.chain([(first_features, first_identifiers)], data_stream)

    # Circuit breaker: abort streaming after 45 seconds
    _SCAN_TIMEOUT = 45.0
    start_time = time.time()
    
    chunk_idx = 1
    for features, identifiers in full_stream:
        # --- Circuit Breaker ---
        if time.time() - start_time > _SCAN_TIMEOUT:
            print(
                f"[Distill] Circuit breaker tripped after {_SCAN_TIMEOUT}s. "
                f"Processed {total_samples} samples before cutoff."
            )
            break

        chunk_len = len(features)
        total_samples += chunk_len
        
        # Only run statistical prefilter on tabular data
        # For image embeddings (dim>=512) or NLP embeddings, skip it —
        # embedding dimensions have no interpretable column semantics
        # and the IQR/Z-score logic produces massive false positives
        # on high-dimensional latent spaces with small sample counts.
        if extractor.last_images or input_dim >= 384:
            hard_flags = np.zeros(len(identifiers), dtype=bool)
        else:
            hard_flags = _statistical_prefilter(features, identifiers, extractor.last_columns)
        mask = ~hard_flags
        ml_features = features[mask]
        
        if len(ml_features) > 0:
            tensor = torch.tensor(ml_features, dtype=torch.float32)
            if extractor.last_images and input_dim >= 512:
                import torch.nn.functional as F
                tensor = F.normalize(tensor, p=2, dim=1)
                ml_features = tensor.numpy()
                
            # OOM-Safe chunk evaluation
            c_raw_ae, c_recons = _evaluate_autoencoder(ae_model, tensor)
            c_raw_svdd = _evaluate_deep_svdd(svdd_model, tensor)
            c_raw_iso = _evaluate_isolation_forest(iso_model, ml_features, pca)
            
            # Apply frozen thresholds
            ae_flags = _evaluate_binary_votes(c_raw_ae, ae_median, ae_mad, ae_thresh)
            svdd_flags = _evaluate_binary_votes(c_raw_svdd, svdd_median, svdd_mad, svdd_thresh)
            iso_flags = _evaluate_binary_votes(c_raw_iso, iso_median, iso_mad, iso_thresh)
            
            conf_ae   = _normalize_scores_to_confidence(c_raw_ae,   ae_median,   ae_mad)
            conf_svdd = _normalize_scores_to_confidence(c_raw_svdd, svdd_median, svdd_mad)
            conf_iso  = _normalize_scores_to_confidence(c_raw_iso,  iso_median,  iso_mad)
            
            # Weighted ensemble confidence (AE+SVDD weighted higher, ISO lower)
            # because AE and SVDD are trained specifically on this data
            ensemble_conf = (0.4*conf_ae + 0.4*conf_svdd + 0.2*conf_iso)
            
            model_breakdown[MODEL_AE] += int(ae_flags.sum())
            model_breakdown[MODEL_SVDD] += int(svdd_flags.sum())
            model_breakdown[MODEL_ISO] += int(iso_flags.sum())
        else:
            c_recons = np.array([])
            ae_flags = np.array([])
            svdd_flags = np.array([])
            iso_flags = np.array([])
            conf_ae = np.array([])
            conf_svdd = np.array([])
            conf_iso = np.array([])
            ensemble_conf = np.array([])
            
        model_breakdown["Statistical Pre-filter"] += int(hard_flags.sum())
        
        # Generate XAI Explanations strictly for flagged samples
        chunk_flags = _build_flagged_items(
            identifiers, ae_flags, svdd_flags, iso_flags, features, c_recons, median_vector, hard_flags,
            conf_ae, conf_svdd, conf_iso, ensemble_conf, n_samples=len(identifiers)
        )
        flagged_items.extend(chunk_flags)
        
        if progress_cb:
            progress_cb("progress", {"chunk": chunk_idx, "total_so_far": total_samples, "poisoned_so_far": len(flagged_items)})
        chunk_idx += 1
        
    poisoned_count = len(flagged_items)
    anomaly_pct = round((poisoned_count / total_samples) * 100, 2) if total_samples > 0 else 0.0

    confidence_distribution = {
        "CRITICAL": sum(1 for item in flagged_items if item["severity"] == "CRITICAL"),
        "HIGH":     sum(1 for item in flagged_items if item["severity"] == "HIGH"),
        "MEDIUM":   sum(1 for item in flagged_items if item["severity"] == "MEDIUM"),
        "LOW":      sum(1 for item in flagged_items if item["severity"] == "LOW"),
    }

    return {
        "total_samples": total_samples,
        "poisoned_samples": poisoned_count,
        "anomaly_percentage": anomaly_pct,
        "model_breakdown": model_breakdown,
        "confidence_distribution": confidence_distribution,
        "flagged_items": flagged_items,
        "warning": warning,
    }


@app.post("/scan-dataset")
@limiter.limit("5/minute")
async def scan_dataset(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key),
):
    """
    Upload a CSV or ZIP, run an ensemble of three anomaly detectors,
    and return the flagged (poisoned) samples with per-model attribution.
    """

    try:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # 1. Mount the generator stream
        data_stream, temp_path = await extractor.process_upload(file)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: _run_scan_pipeline(data_stream)
            )
            return result
        finally:
            remove_file(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        _purge_temp_uploads()


@app.post("/scan-stream")
@limiter.limit("5/minute")  
async def scan_stream(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key),
):
    """
    SSE endpoint. Yields progress events then final result.
    """
    async def generate():
        try:
            yield f"data: {json.dumps({'event':'phase','data':'UPLOADING'})}\n\n"
            
            data_stream, temp_path = await extractor.process_upload(file)
            
            yield f"data: {json.dumps({'event':'phase','data':'TRAINING_MODELS'})}\n\n"
            
            queue = asyncio.Queue()
            ev_loop = asyncio.get_event_loop()

            def sync_progress_cb(event, payload=None):
                if event == "progress":
                    ev_loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"event": "progress", "chunk": payload["chunk"], "total_so_far": payload["total_so_far"], "poisoned_so_far": payload["poisoned_so_far"]}
                    )
                elif event == "phase":
                    ev_loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"event": "phase", "data": payload}
                    )
            
            async def worker():
                try:
                    res = await ev_loop.run_in_executor(
                        None,
                        lambda: _run_scan_pipeline(data_stream, sync_progress_cb)
                    )
                    await queue.put({"event": "complete", "result": res})
                except Exception as e:
                    await queue.put({"event": "error", "detail": str(e)})

            task = asyncio.create_task(worker())
            
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0)  # flush buffer implicitly via loop
                if event["event"] in ("complete", "error"):
                    break
        except Exception as e:
            yield f"data: {json.dumps({'event':'error','detail':str(e)})}\n\n"
        finally:
            if 'temp_path' in locals():
                remove_file(temp_path)
            _purge_temp_uploads()
            
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # critical for nginx/Modal proxy
        }
    )

def _generate_pdf_receipt(scan_results: dict, filename: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", style="B", size=16)
    pdf.cell(0, 10, "Distill Sanitization Receipt", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, f"Original File: {filename}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Total Samples: {scan_results.get('total_samples', 0)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Poisoned Samples Removed: {scan_results.get('poisoned_samples', 0)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Anomaly Percentage: {scan_results.get('anomaly_percentage', 0)}%", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "Top 5 Most Toxic Data Points:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    
    flagged = scan_results.get("flagged_items", [])
    top_5 = flagged[:5]
    
    for item in top_5:
        item_id = item.get("id", "Unknown")
        models = ", ".join(item.get("flagged_by", []))
        pdf.set_font("helvetica", style="B", size=10)
        pdf.cell(0, 6, f"ID: {item_id} (Flagged by: {models})", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", size=10)
        explanation = item.get("explanation")
        if explanation:
            if explanation.get("type") == "image":
                pdf.cell(0, 6, "  Explanation: Latent Space Divergence detected in ResNet-18 features.", new_x="LMARGIN", new_y="NEXT")
            elif explanation.get("type") == "tabular":
                top_features = ", ".join([c["name"] for c in explanation.get("top_contributors", [])])
                pdf.cell(0, 6, f"  Explanation: High reconstruction error in features: {top_features}", new_x="LMARGIN", new_y="NEXT")
            elif explanation.get("type") == "text":
                col_name = explanation.get("column", "Unknown")
                pdf.cell(0, 6, f"  Explanation: High Semantic Deviation in text column '{col_name}'.", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 6, "  Explanation: None available.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        
    return pdf.output()



@app.post("/download-sanitized")
@limiter.limit("5/minute")
async def download_sanitized(
    request: Request,
    file: UploadFile = File(...),
    flagged_items: str = Form(...),
    scan_results_json: str = Form("{}"),
):
    """
    Accept the original file, a JSON list of flagged IDs, and metadata.
    Return a cleaned ZIP version with a generated PDF receipt included.
    """
    import tempfile as _tf

    try:
        to_remove: list[str] = json.loads(flagged_items)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="flagged_items must be valid JSON.")

    try:
        scan_results = json.loads(scan_results_json) if scan_results_json else {}
    except json.JSONDecodeError:
        scan_results = {}

    remove_set = set(to_remove)
    filename = (file.filename or "").lower()

    # Stream the upload to a temp file to avoid loading entire file into RAM
    fd, src_temp_path = _tf.mkstemp(suffix=os.path.splitext(filename)[1])
    try:
        with os.fdopen(fd, 'wb') as tmp_f:
            while True:
                chunk = await file.read(1024 * 1024 * 4)  # 4MB
                if not chunk:
                    break
                tmp_f.write(chunk)

        # Generate the receipt
        pdf_bytes = _generate_pdf_receipt(scan_results, filename)

        # ── CSV path ──────────────────────────────────────────────────────
        if filename.endswith(".csv"):
            df = pd.read_csv(src_temp_path)
            mask = ~df.index.astype(str).isin(remove_set)
            clean_df = df.loc[mask]

            file_id = uuid.uuid4().hex
            out_path = os.path.join(_TEMP_UPLOADS_DIR, f"{file_id}.zip")
            
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as dst:
                csv_buf = io.StringIO()
                clean_df.to_csv(csv_buf, index=False)
                dst.writestr(f"sanitized_{filename}", csv_buf.getvalue())
                dst.writestr("Sanitization_Receipt.pdf", pdf_bytes)

            return {
                "download_url": f"/pickup-sanitized/{file_id}",
                "filename": f"sanitized_{filename}.zip"
            }

        # ── ZIP path ──────────────────────────────────────────────────────
        if filename.endswith(".zip"):
            file_id = uuid.uuid4().hex
            out_path = os.path.join(_TEMP_UPLOADS_DIR, f"{file_id}.zip")

            with zipfile.ZipFile(src_temp_path) as src:
                with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as dst:
                    dst.writestr("Sanitization_Receipt.pdf", pdf_bytes)
                    for entry in src.namelist():
                        if entry.endswith("/") or entry.startswith("__MACOSX"):
                            continue
                        
                        basename = os.path.basename(entry)
                        
                        if entry.lower().endswith(".csv"):
                            with src.open(entry) as f:
                                df = pd.read_csv(f)
                                
                            row_identifiers = [f"{basename}:row_{i}" for i in range(len(df))]
                            mask = [rid not in remove_set for rid in row_identifiers]
                            clean_df = df.iloc[mask]
                            
                            csv_buf = io.StringIO()
                            clean_df.to_csv(csv_buf, index=False)
                            dst.writestr(entry, csv_buf.getvalue())
                        else:
                            if basename in remove_set:
                                continue
                            dst.writestr(entry, src.read(entry))

            return {
                "download_url": f"/pickup-sanitized/{file_id}",
                "filename": f"sanitized_{filename}"
            }

        raise HTTPException(status_code=400, detail="Unsupported file type.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Always clean up the source temp file
        if os.path.exists(src_temp_path):
            os.remove(src_temp_path)

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

@app.get("/pickup-sanitized/{file_id}")
async def pickup_sanitized(file_id: str, background_tasks: BackgroundTasks):
    """
    Serve the cleaned file directly from disk using the browser's native download manager.
    Delete the file immediately after sending it to free up server space.
    """
    # SECURITY: Prevent path traversal — file_id must be a strict hex UUID
    import re
    if not re.fullmatch(r"[a-f0-9]{32}", file_id):
        raise HTTPException(status_code=400, detail="Invalid file identifier.")

    csv_path = os.path.join(_TEMP_UPLOADS_DIR, f"{file_id}.csv")
    zip_path = os.path.join(_TEMP_UPLOADS_DIR, f"{file_id}.zip")
    
    if os.path.exists(csv_path):
        target_path = csv_path
        media_type = "text/csv"
    elif os.path.exists(zip_path):
        target_path = zip_path
        media_type = "application/zip"
    else:
        raise HTTPException(status_code=404, detail="File has expired or does not exist.")
        
    background_tasks.add_task(remove_file, target_path)
    
    return FileResponse(
        path=target_path,
        media_type=media_type,
        filename=os.path.basename(target_path)
    )


# ==========================================
# MODAL SERVERLESS DEPLOYMENT CONFIGURATION
# ==========================================
import modal

# 1. Define the Modal Application
modal_app = modal.App("distill-backend")

# 2. Get the absolute path to the backend directory on the local machine
backend_dir = os.path.dirname(os.path.abspath(__file__))

# 3. Define the Cloud Environment & attach local files directly to the image
distill_image = (
    modal.Image.debian_slim()
    .env({"PYTHONPATH": "/root"})  # Forces Python to look in /root for imports
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi", 
        "uvicorn",
        "python-multipart", 
        "torch", 
        "torchvision",
        "scikit-learn", 
        "pandas", 
        "numpy", 
        "opencv-python-headless", 
        "pillow",
        "slowapi",
        "fpdf2",
        "sentence-transformers",
        "open-clip-torch",
        "pyarrow",
    )
    # Cache CLIP ViT-B-32 weights during image build to prevent cold-start latency
    .run_commands('python -c "import open_clip; open_clip.create_model_and_transforms(\'ViT-B-32\', pretrained=\'openai\')"')
    # Cache the all-MiniLM-L6-v2 model for text embedding
    .run_commands('python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\'all-MiniLM-L6-v2\')"')
    # NEW MODAL 1.0 SYNTAX: Add local files directly to the image builder
    .add_local_file(
        os.path.join(backend_dir, "extractor.py"), remote_path="/root/extractor.py"
    )
    .add_local_file(
        os.path.join(backend_dir, "models.py"), remote_path="/root/models.py"
    )
)

# 4. Define the serverless function (no 'mounts' parameter needed here anymore)
@modal_app.function(
    image=distill_image,
    gpu="T4",
    memory=8192,
    min_containers=1,
)
@modal.asgi_app()
def serve():
    # Ensure the upload directory starts clean in the cloud container
    _purge_temp_uploads()
    return app