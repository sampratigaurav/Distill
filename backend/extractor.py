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
import tempfile
import ast
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
        self._text_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self.last_columns: List[str] = []
        self.last_images: dict[str, Image.Image] = {}
        self.current_chunk_text_snippets: dict[str, dict[str, str]] = {}

    def _load_text_model(self) -> None:
        if self._text_model is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from sentence_transformers import SentenceTransformer
            self._text_model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)

    def _load_clip(self) -> None:
        if self._clip_model is None:
            import open_clip
            self._clip_model, _, self._clip_preprocess = (
                open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='openai'
                )
            )
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_upload(
        self, file: UploadFile
    ) -> Tuple[object, str]:
        """
        Save file to temp storage and return (generator_stream, temp_file_path).
        """
        filename = (file.filename or "").lower()
        fd, temp_path = tempfile.mkstemp(suffix=_file_ext(filename))
        
        with os.fdopen(fd, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024 * 4) # 4MB
                if not chunk:
                    break
                f.write(chunk)
                
        if filename.endswith(".csv"):
            return self._stream_csv(temp_path), temp_path
        elif filename.endswith(".zip"):
            return self._stream_zip(temp_path), temp_path
        elif filename.endswith(".parquet"):
            return self._stream_parquet(temp_path), temp_path
        else:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type: Please upload a .csv, .parquet, or .zip archive.",
            )

    # ------------------------------------------------------------------
    # Tabular (DataFrame) pipeline
    # ------------------------------------------------------------------

    def _stream_csv(self, file_path: str, id_prefix: str = ""):
        self.last_images.clear()
        self.last_columns.clear()
        self.current_chunk_text_snippets.clear()

        # Track state for scaling across chunks
        num_cols = None
        cat_cols = None
        text_cols = None
        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        mean_vals = None
        cat_fill_vals = None
        
        chunk_idx = 0
        
        # Read file in 10,000 row chunks strictly from disk
        for df in pd.read_csv(file_path, chunksize=MAX_SAMPLE_SIZE):
            
            # FIT PHASE: Chunk 0 calculates state
            if chunk_idx == 0:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                str_cols = df.select_dtypes(exclude="number").columns.tolist()
                
                # Check if this is a Zip combined file which has __source_file__
                if "__source_file__" in num_cols: num_cols.remove("__source_file__")
                if "__source_file__" in str_cols: str_cols.remove("__source_file__")

                cat_cols = []
                text_cols = []
                
                for col in str_cols:
                    if df[col].nunique() > 100:
                        avg_len = df[col].astype(str).str.len().mean()
                        if avg_len > 15:
                            text_cols.append(col)
                        else:
                            cat_cols.append(col)
                    else:
                        cat_cols.append(col)

                if num_cols:
                    mean_vals = df[num_cols].mean()
                    num_train = df[num_cols].fillna(mean_vals).fillna(0.0)
                    scaler.fit(num_train.values)
                    self.last_columns.extend(num_cols)
                    
                if cat_cols:
                    cat_fill_vals = {}
                    cat_dict_train = {}
                    valid_cat_cols = []
                    for col in cat_cols:
                        if df[col].nunique() > 100: continue
                        mode_val = df[col].mode()
                        fill_val = mode_val.iloc[0] if not mode_val.empty else "MISSING"
                        cat_fill_vals[col] = fill_val
                        cat_dict_train[col] = df[col].fillna(fill_val)
                        valid_cat_cols.append(col)
                        
                    cat_cols = valid_cat_cols
                    
                    if cat_dict_train:
                        cat_df = pd.DataFrame(cat_dict_train)
                        encoder.fit(cat_df.values)
                        self.last_columns.extend(list(encoder.get_feature_names_out(cat_df.columns)))
                        
                if text_cols:
                    self._load_text_model()
                    for col in text_cols:
                        for dim in range(384): # all-MiniLM-L6-v2 dim
                            self.last_columns.append(f"__TEXT__:{col}_dim{dim}")

            # YIELD PHASE: transform current chunk
            if "__source_file__" in df.columns:
                 source_col = df["__source_file__"]
                 identifiers = [f"{source_col.iloc[i]}:row_{chunk_idx*MAX_SAMPLE_SIZE + i}" for i in range(len(df))]
            else:
                 identifiers = [f"{id_prefix}{chunk_idx*MAX_SAMPLE_SIZE + i}" for i in range(len(df))]
                 
            self.current_chunk_text_snippets.clear()
            parts = []
            
            if num_cols:
                num_chunk = df[num_cols].fillna(mean_vals).fillna(0.0)
                parts.append(scaler.transform(num_chunk.values))
                
            if cat_cols:
                cat_dict_chunk = {}
                for col in cat_cols:
                    cat_dict_chunk[col] = df[col].fillna(cat_fill_vals[col])
                cat_df_chunk = pd.DataFrame(cat_dict_chunk)
                if cat_df_chunk.shape[1] > 0:
                    parts.append(encoder.transform(cat_df_chunk.values))
                    
            if text_cols:
                self._load_text_model()
                for col in text_cols:
                    text_data = df[col].fillna("").astype(str).tolist()
                    
                    for i, ident in enumerate(identifiers):
                        if ident not in self.current_chunk_text_snippets:
                            self.current_chunk_text_snippets[ident] = {}
                        self.current_chunk_text_snippets[ident][col] = text_data[i][:500] # store max 500 chars

                    encode_bs = 512 if torch.cuda.is_available() else 256
                    embeddings = self._text_model.encode(text_data, batch_size=encode_bs, show_progress_bar=False, normalize_embeddings=True)
                    parts.append(embeddings)
                
            if not parts:
                raise HTTPException(status_code=400, detail="Contains no usable columns.")
                
            features = np.concatenate(parts, axis=1).astype(np.float32)
                 
            yield features, identifiers
            chunk_idx += 1

    def _stream_parquet(self, file_path: str):
        """Convert a Parquet file to a temp CSV and reuse the CSV pipeline."""
        import pyarrow.parquet as pq
        import tempfile
        table = pq.read_table(file_path)
        df = table.to_pandas()

        fd, csv_path = tempfile.mkstemp(suffix='.csv')
        try:
            with os.fdopen(fd, 'w') as f:
                df.to_csv(f, index=False)
            yield from self._stream_csv(csv_path)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


    # ZIP pipeline (content-aware)
    # ------------------------------------------------------------------

    def _stream_zip(self, temp_path: str):
        with zipfile.ZipFile(temp_path) as archive:
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

            if csv_entries:
                yield from self._stream_zip_csvs(archive, csv_entries)
            elif img_entries:
                yield from self._stream_zip_images(archive, img_entries)
            else:
                raise HTTPException(status_code=400, detail="The archive contains no supported files.")

    # ---- ZIP → tabular ---------------------------------------------------

    def _stream_zip_csvs(self, archive: zipfile.ZipFile, csv_entries: list[str]):
        """Safely stream all CSVs by building a merged temporary dataset."""
        _MAX_UNCOMPRESSED = 1073741824  # 1 GB strict limit across all files
        running_size = 0
        
        fd, combined_path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as combined_f:
                header_written = False
                for entry in csv_entries:
                    with archive.open(entry) as f:
                        buf = bytearray()
                        while True:
                            chunk = f.read(8192)
                            if not chunk: break
                            running_size += len(chunk)
                            if running_size > _MAX_UNCOMPRESSED:
                                raise HTTPException(413, "Uncompressed contents exceed 1GB physical limit.")
                            buf.extend(chunk)
                            
                    df = pd.read_csv(io.BytesIO(buf))
                    basename = os.path.basename(entry)
                    df["__source_file__"] = basename
                    df.to_csv(combined_f, index=False, header=not header_written)
                    header_written = True
                    
            yield from self._stream_csv(combined_path)
            
        finally:
            if os.path.exists(combined_path):
                os.remove(combined_path)

    # ---- ZIP → images ----------------------------------------------------

    def _load_resnet(self) -> None:
        """Lazy-init a headless ResNet-18 feature extractor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        resnet.eval()
        
        self._resnet = resnet.to(self.device)

        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _stream_zip_images(self, archive: zipfile.ZipFile, img_entries: list[str]):
        """Embed images via CLIP ViT-B-32 and yield chunked 512-dim feature vectors."""
        if self._clip_model is None:
            self._load_clip()

        _MAX_UNCOMPRESSED = 1073741824  # 1 GB strict limit
        running_size = 0
        pil_images: list[Image.Image] = []
        filenames: list[str] = []

        self.last_columns.clear()
        self.last_images.clear()

        chunk_size = 128

        for entry in img_entries:
            with archive.open(entry) as f:
                buf = bytearray()
                while True:
                    chunk = f.read(8192)
                    if not chunk: break
                    running_size += len(chunk)
                    if running_size > _MAX_UNCOMPRESSED:
                        raise HTTPException(413, "Physical limits exceeded.")
                    buf.extend(chunk)

            img = Image.open(io.BytesIO(buf)).convert("RGB")
            filename = os.path.basename(entry)
            self.last_images[filename] = img.copy()

            pil_images.append(img)
            filenames.append(filename)

            if len(pil_images) >= chunk_size:
                batch_tensor = torch.stack(
                    [self._clip_preprocess(img) for img in pil_images]
                ).to(self.device)
                with torch.no_grad():
                    features = self._clip_model.encode_image(batch_tensor)
                    features = features.cpu().float().numpy()
                yield features, filenames

                # Clear per-chunk memory to stay OOM-safe
                self.last_images.clear()
                pil_images = []
                filenames = []

        if pil_images:
            batch_tensor = torch.stack(
                [self._clip_preprocess(img) for img in pil_images]
            ).to(self.device)
            with torch.no_grad():
                features = self._clip_model.encode_image(batch_tensor)
                features = features.cpu().float().numpy()
            yield features, filenames
