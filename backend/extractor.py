"""
Universal feature extractor for the Data Sanitization API.

Accepts an uploaded file (CSV or ZIP) and returns a normalised
2-D NumPy feature matrix together with human-readable identifiers for
every row.

ZIP handling is content-aware:
  • If the archive contains .csv files → tabular pipeline
  • If it contains images (.jpg, .jpeg, .png) → ResNet-18 embeddings
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from fastapi import HTTPException, UploadFile
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Supported extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_CSV_EXTENSIONS = {".csv"}

# Maximum number of rows processed by the tabular pipeline.
# Datasets larger than this are randomly down-sampled before feature
# extraction to keep training times in the single-digit-second range.
MAX_SAMPLE_SIZE = 10000


def _file_ext(name: str) -> str:
    """Return the lowercase extension including the leading dot."""
    return ("." + name.rsplit(".", maxsplit=1)[-1].lower()) if "." in name else ""


class UniversalExtractor:
    """
    Stateless helper that turns an arbitrary upload into a feature matrix.

    Usage::

        extractor = UniversalExtractor()
        features, identifiers = await extractor.process_upload(file)
    """

    def __init__(self) -> None:
        # Lazy-loaded on first image ZIP upload
        self._resnet: nn.Module | None = None
        self._transform: T.Compose | None = None
        self.last_columns: List[str] = []
        self.last_images: dict[str, Image.Image] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_upload(
        self, file: UploadFile
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Read *file* and return ``(features, identifiers)``.

        Parameters
        ----------
        file : fastapi.UploadFile
            An uploaded ``.csv`` or ``.zip`` (containing CSVs **or** images).

        Returns
        -------
        features : np.ndarray
            2-D array of shape ``(n_samples, n_features)``.
        identifiers : list[str]
            Row indices (CSV) or filenames (images) — one per sample.

        Raises
        ------
        HTTPException (400)
            If the file type is not supported.
        """
        filename = (file.filename or "").lower()

        if filename.endswith(".csv"):
            raw = await file.read()
            df = pd.read_csv(io.BytesIO(raw))
            return self._process_dataframe(df)
        elif filename.endswith(".zip"):
            return await self._process_zip(file)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: '{filename}'. "
                    "Please upload a .csv or a .zip archive."
                ),
            )

    # ------------------------------------------------------------------
    # Tabular (DataFrame) pipeline
    # ------------------------------------------------------------------

    def _process_dataframe(
        self, df: pd.DataFrame, id_prefix: str = ""
    ) -> Tuple[np.ndarray, List[str]]:
        """Impute, scale & encode a DataFrame. Returns features + identifiers."""
        self.last_images.clear()
        self.last_columns.clear()

        # Down-sample large datasets to keep training times bounded.
        if len(df) > MAX_SAMPLE_SIZE:
            df = df.sample(n=MAX_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        parts: list[np.ndarray] = []

        # --- Numerical columns ---
        if num_cols:
            num_df = df[num_cols].copy()
            # Handle partial missing values with column mean, and all-NaN cols with 0.0
            num_df = num_df.fillna(num_df.mean()).fillna(0.0)
            scaler = StandardScaler()
            parts.append(scaler.fit_transform(num_df.values))
            self.last_columns.extend(num_cols)

        # --- Categorical columns ---
        if cat_cols:
            cat_dict = {}
            for col in cat_cols:
                col_series = df[col]
                if col_series.nunique() > 100:
                    continue
                mode_val = col_series.mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "MISSING"
                cat_dict[col] = col_series.fillna(fill_val)
                
            if cat_dict:
                cat_df = pd.DataFrame(cat_dict)
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                parts.append(encoder.fit_transform(cat_df.values))
                self.last_columns.extend(list(encoder.get_feature_names_out(cat_df.columns)))

        if not parts:
            raise HTTPException(
                status_code=400,
                detail="The CSV contains no usable columns.",
            )

        features = np.concatenate(parts, axis=1).astype(np.float32)
        identifiers = [f"{id_prefix}{idx}" for idx in df.index.tolist()]
        return features, identifiers

    # ------------------------------------------------------------------
    # ZIP pipeline (content-aware)
    # ------------------------------------------------------------------

    async def _process_zip(
        self, file: UploadFile
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Inspect ZIP contents and route to the right pipeline:
          • CSV files  → tabular pipeline (concatenated row-wise)
          • Image files → ResNet-18 embedding pipeline
        """
        raw = await file.read()
        archive = zipfile.ZipFile(io.BytesIO(raw))

        # --- Zip bomb protection: reject archives whose total uncompressed
        #     size exceeds 1 GB before extracting any content. ---
        _MAX_UNCOMPRESSED = 1 * 1024 * 1024 * 1024  # 1 GB
        total_uncompressed = sum(info.file_size for info in archive.infolist())
        if total_uncompressed > _MAX_UNCOMPRESSED:
            raise HTTPException(
                status_code=413,
                detail="ZIP archive uncompressed size exceeds the 1 GB limit.",
            )

        # Classify entries
        csv_entries: list[str] = []
        img_entries: list[str] = []

        for entry in archive.namelist():
            if entry.endswith("/") or entry.startswith("__MACOSX"):
                continue
            ext = _file_ext(entry)
            if ext in _CSV_EXTENSIONS:
                csv_entries.append(entry)
            elif ext in _IMAGE_EXTENSIONS:
                img_entries.append(entry)

        # Decide pipeline
        if csv_entries:
            return self._process_zip_csvs(archive, csv_entries)
        elif img_entries:
            return self._process_zip_images(archive, img_entries)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The ZIP archive contains no supported files. "
                    "Include .csv files or images (.jpg, .jpeg, .png)."
                ),
            )

    # ---- ZIP → tabular ---------------------------------------------------

    def _process_zip_csvs(
        self, archive: zipfile.ZipFile, csv_entries: list[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Read every CSV inside the archive, concat, and run the tabular pipeline."""
        frames: list[pd.DataFrame] = []
        for entry in csv_entries:
            with archive.open(entry) as f:
                df = pd.read_csv(f)
            # Tag rows with the source filename (sanitized to prevent path traversal)
            basename = os.path.basename(entry)
            df.index = pd.RangeIndex(len(df))
            df["__source_file__"] = basename
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)

        # Build identifiers from source file + row offset
        source_col = combined.pop("__source_file__")
        identifiers_raw = [
            f"{source_col.iloc[i]}:row_{i}" for i in range(len(combined))
        ]

        features, _ = self._process_dataframe(combined)
        return features, identifiers_raw

    # ---- ZIP → images ----------------------------------------------------

    def _load_resnet(self) -> None:
        """Lazy-init a headless ResNet-18 feature extractor."""
        # --- SAFE FIX: Detect device and move model to GPU if available ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        resnet.eval()
        
        self._resnet = resnet.to(self.device)
        # ------------------------------------------------------------------

        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _process_zip_images(
        self, archive: zipfile.ZipFile, img_entries: list[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Embed images via ResNet-18 and return 512-dim feature vectors."""
        if self._resnet is None:
            self._load_resnet()

        tensors: list[torch.Tensor] = []
        filenames: list[str] = []
        
        self.last_columns.clear()
        self.last_images.clear()

        for entry in img_entries:
            img_bytes = archive.read(entry)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Sanitize filename to prevent path traversal
            filename = os.path.basename(entry)
            self.last_images[filename] = img.copy()
            
            tensor = self._transform(img)  # type: ignore[misc]
            tensors.append(tensor)
            filenames.append(filename)

        batch = torch.stack(tensors)

        # --- SAFE FIX: Process in chunks to prevent OOM ---
        embeddings_list = []
        chunk_size = 64
        
        with torch.no_grad():
            for i in range(0, len(batch), chunk_size):
                # --- SAFE FIX: Move chunk to GPU before passing to ResNet ---
                chunk = batch[i:i + chunk_size].to(self.device)
                chunk_embeds = self._resnet(chunk)
                embeddings_list.append(chunk_embeds)
                # ------------------------------------------------------------
                
        embeddings = torch.cat(embeddings_list, dim=0)
        # --------------------------------------------------

        features = embeddings.cpu().numpy().astype(np.float32)
        return features, filenames
