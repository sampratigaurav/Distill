<div align="center">

<br/>

```
██████╗ ██╗███████╗████████╗██╗██╗     ██╗
██╔══██╗██║██╔════╝╚══██╔══╝██║██║     ██║
██║  ██║██║███████╗   ██║   ██║██║     ██║
██║  ██║██║╚════██║   ██║   ██║██║     ██║
██████╔╝██║███████║   ██║   ██║███████╗███████╗
╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝╚══════╝
```

**Universal Data Sanitization & Poisoning Detection**

[![CI](https://github.com/sampratigaurav/Distill/actions/workflows/ci.yml/badge.svg)](https://github.com/sampratigaurav/Distill/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-orange?logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML%20Engine-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Modal](https://img.shields.io/badge/Deployed%20on-Modal-6366f1?logo=modal&logoColor=white)](https://modal.com)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?logo=vercel&logoColor=white)](https://vercel.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-f97316?logoColor=white)](LICENSE)

<br/>

> *Distill bridges the gap between raw data collection and model training — a deterministic, multi-model defense layer that mathematically hunts down poisoned data at any scale.*

<br/>

[**Live Demo →**](https://distill-nine-theta.vercel.app) &nbsp;&nbsp;|&nbsp;&nbsp; [**API Docs →**](https://distill-nine-theta.vercel.app/docs) &nbsp;&nbsp;|&nbsp;&nbsp; [**Report Bug →**](https://github.com/sampratigaurav/Distill/issues)

<br/>

</div>

---

## What is Distill?

Distill is a high-performance, **unsupervised machine learning suite** designed to identify and quarantine anomalous or "poisoned" data inside massive datasets — before they reach your model.

Whether you're uploading a CSV of network logs, a ZIP of training images, or a dataset of free-text reviews, Distill automatically translates your data into a universal latent space and flags statistical outliers using a **2-of-3 ensemble voting system**.

```
[ Raw Dataset ] → [ Feature Extraction ] → [ Ensemble Detection ] → [ Clean Output + PDF Receipt ]
     CSV / ZIP      Tabular · Image · NLP     AE · DeepSVDD · ISO       Sanitized dataset + report
```

---

## Feature Highlights

### 🧠 Universal Feature Extraction

| Data Type | Pipeline |
|-----------|----------|
| **Tabular / CSV** | Pandas scaling + OneHot encoding via scikit-learn |
| **Images (ZIP)** | **CLIP ViT-B/32** → 512-dim embedding |
| **Free Text** | **SentenceTransformer** (`all-MiniLM-L6-v2`) → 384-dim semantic vector |

Distill auto-detects column types and routes your data through the correct extraction pipeline — no configuration required.

---

### 🛡️ Ensemble Defense — Strict Democracy

Three independently trained anomaly detectors vote on every sample. A sample is flagged **only when ≥ 2 of 3 models agree**.

```
Sample
  ├──▶ Autoencoder         (Reconstruction Error)   ──┐
  ├──▶ Deep SVDD           (Hypersphere Distance)   ──┼──▶ ≥2 votes → POISONED
  └──▶ Isolation Forest    (Decision Function)      ──┘
```

This strict 2-of-3 consensus dramatically reduces false positives on clean, high-variance datasets.

---

### 📊 Explainable AI (XAI)

Distill moves beyond black-box anomaly detection. For every flagged sample, it surfaces *why* it was flagged:

- **Tabular** → Top 3 columns by absolute reconstruction error
- **Images** → Top 10 ResNet-18 latent dimensions by deviation from median
- **Text** → Exact text snippet causing high semantic deviation, with column name

---

### 🔥 Streaming Architecture

Distill streams data in memory-safe chunks (`10,000 rows` per chunk for tabular, `64 images` per batch). MAD thresholds are **calibrated once on Chunk 0 and frozen**, ensuring deterministic scores across all downstream chunks without re-training.

```
Chunk 0 → Train Models → Calibrate MAD Thresholds → Freeze
Chunk 1 → Evaluate  ──┐
Chunk 2 → Evaluate  ──┤──▶ Apply frozen thresholds → Flag
...     → Evaluate  ──┘
```

---

## Security Architecture

Distill is engineered for adversarial environments:

| Threat | Defense |
|--------|---------|
| **HTTP Header Spoofing / Memory DoS** | ASGI byte-counting middleware — physically counts streamed bytes, enforces 1 GB hard limit regardless of `Content-Length` header |
| **Zip Bomb** | Streaming x-ray extraction — halts the moment uncompressed bytes exceed 1 GB, ignoring forged metadata |
| **Path Traversal on Download** | `file_id` validated against strict `[a-f0-9]{32}` regex before any file system access |
| **Abuse / Rate Limiting** | SlowAPI enforces 5 requests/minute per IP on heavy ML endpoints |
| **Browser Memory Crash (large downloads)** | Two-step "pickup box" — cleaned ZIP written to server disk, served via `FileResponse` (native browser download manager), deleted immediately after |

---

## Tech Stack

### Backend — The ML Engine

| Component | Technology |
|-----------|------------|
| Framework | **FastAPI** + Uvicorn |
| Deep Learning | **PyTorch** — DynamicAutoencoder, DynamicDeepSVDD |
| Computer Vision | **OpenCLIP** ViT-B/32 |
| NLP Embeddings | **SentenceTransformers** `all-MiniLM-L6-v2` |
| Classical ML | **scikit-learn** IsolationForest |
| Data Processing | **Pandas**, NumPy |
| PDF Reports | **FPDF2** |
| Rate Limiting | **SlowAPI** |
| Serverless Deployment | **Modal** (T4 GPU, 8 GB RAM) |

### Frontend — The Dashboard

| Component | Technology |
|-----------|------------|
| Framework | **Next.js 16** (App Router) |
| Language | **TypeScript** |
| Styling | **Tailwind CSS v4** |
| Charts | **Recharts** |
| HTTP Client | **Axios** |
| Icons | **Lucide React** |
| Deployment | **Vercel** |

---

## Getting Started

### Prerequisites

- **Python ≥ 3.11**
- **Node.js ≥ 20**

### 1 — Clone the Repository

```bash
git clone https://github.com/sampratigaurav/Distill.git
cd Distill
```

### 2 — Backend (FastAPI + PyTorch)

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

> The backend will be available at `http://localhost:8000`
> Interactive API docs at `http://localhost:8000/docs`

### 3 — Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

> The dashboard will be available at `http://localhost:3000`

### 4 — Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, set `ALLOWED_ORIGINS` in the backend environment:

```env
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

---

## API Reference

### `GET /health`

Liveness / readiness probe.

```json
{ "status": "healthy", "version": "0.2.0" }
```

---

### `POST /scan-dataset`

Upload a `.csv` or `.zip` file for ensemble anomaly detection.

**Rate limit:** 5 requests/minute per IP

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `File` | `.csv` or `.zip` archive |

**Response:**

```json
{
  "total_samples": 10000,
  "poisoned_samples": 47,
  "anomaly_percentage": 0.47,
  "model_breakdown": {
    "Autoencoder": 52,
    "Deep SVDD": 61,
    "Isolation Forest": 48
  },
  "flagged_items": [
    {
      "id": "row_42",
      "flagged_by": ["Autoencoder", "Deep SVDD"],
      "explanation": {
        "type": "tabular",
        "top_contributors": [
          { "name": "packet_size", "error": 4.821 },
          { "name": "duration_ms", "error": 3.107 },
          { "name": "dst_port", "error": 1.923 }
        ]
      }
    }
  ]
}
```

---

### `POST /download-sanitized`

Upload the original file + scan results. Returns a cleaned ZIP with a PDF sanitization receipt.

**Rate limit:** 5 requests/minute per IP

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `File` | Original `.csv` or `.zip` |
| `flagged_items` | `string` | JSON array of flagged IDs from `/scan-dataset` |
| `scan_results_json` | `string` | Full scan result JSON for the PDF receipt |

**Response:**

```json
{
  "download_url": "/pickup-sanitized/a3f8d...e21b",
  "filename": "sanitized_dataset.csv.zip"
}
```

---

### `GET /pickup-sanitized/{file_id}`

Serve the cleaned file via the browser's native download manager. The file is deleted from the server immediately after delivery.

---

## ML Architecture Deep Dive

### DynamicAutoencoder

A symmetric encoder–decoder whose layer widths scale automatically with `input_dim`:

```
input_dim → h1 (÷2) → h2 (÷4) → bottleneck (÷8) → h2 → h1 → input_dim
```

Each layer is wrapped with `BatchNorm1d` + `ReLU`. Anomaly score = per-sample MSE reconstruction error.

### DynamicDeepSVDD

Implements [Ruff et al., ICML 2018](https://proceedings.mlr.press/v80/ruff18a.html). The encoder maps inputs into a compact hypersphere. The anomaly score is squared L2 distance from the learned center `c`.

```
input_dim → h1 → h2 → latent_dim
                         ↓
               dist² from center c  ──▶  anomaly score
```

The hypersphere center `c` is initialised as the mean of all encoder outputs on the training chunk before optimisation begins.

### MAD Thresholding

All three models use **Modified Z-Score** thresholding (Iglewicz & Hoaglin, 1993), which is robust to the non-Gaussian distributions common in real-world anomaly detection:

```
Modified Z-Score = 0.6745 × (score − median) / MAD
```

A variance floor of `1e-5` prevents division by zero on uniform datasets. The threshold adapts to dataset size:

| Dataset size | Threshold |
|---|---|
| `n < 200` | `3.0` (more sensitive) |
| `n ≥ 200` | `3.5` (more specific) |

---

## CI / CD Pipeline

```
Push / PR
   │
   ▼
pytest (backend math)
   │  ✅
   ├──▶ [main branch only] Deploy → Vercel (frontend)
   └──▶ [main branch only] Deploy → Modal (backend, T4 GPU)
```

The test suite covers:

- MAD thresholding edge cases (uniform data, small datasets, single/multiple outliers)
- `DynamicAutoencoder` forward pass, reconstruction error shape & non-negativity
- `DynamicDeepSVDD` forward pass, anomaly score shape & non-negativity
- Ensemble 2-of-3 voting correctness for all 8 vote combinations
- Isolation Forest contamination formula bounds

---

## Running Tests

```bash
cd backend
pip install pytest
pytest tests/ -v --tb=short
```

---

## Project Structure

```
Distill/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD: test → deploy (Vercel + Modal)
│
├── backend/
│   ├── app.py                  # FastAPI entry point, ML pipeline, endpoints
│   ├── extractor.py            # Universal feature extraction (tabular/image/text)
│   ├── models.py               # DynamicAutoencoder + DynamicDeepSVDD (PyTorch)
│   ├── requirements.txt
│   ├── Dockerfile              # Hugging Face Spaces / Docker deployment
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py         # Mocks `modal` for CI
│       └── test_math.py        # Unit tests for core ML logic
│
└── frontend/
    ├── app/
    │   ├── page.tsx            # Single-page dashboard (upload → scan → results)
    │   ├── layout.tsx
    │   └── globals.css         # Industrial monochrome design system
    ├── package.json
    ├── next.config.ts
    └── tsconfig.json
```

---

## Deployment

### Backend → Modal (Serverless GPU)

```bash
cd backend
pip install modal
modal deploy app.py::modal_app
```

The Modal image pre-caches both CLIP ViT-B/32 weights and `all-MiniLM-L6-v2` during build to eliminate cold-start latency.

### Frontend → Vercel

```bash
cd frontend
npx vercel --prod
```

Set the following environment variables in your Vercel project settings:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | Your Modal deployment URL |

---

## Roadmap

- [ ] Streaming progress via Server-Sent Events (SSE)
- [ ] Per-model confidence scores alongside binary votes
- [ ] Support for Parquet and JSON Lines datasets
- [ ] Authentication layer for multi-tenant deployments
- [ ] Historical scan comparison dashboard
- [ ] REST webhook notifications on scan completion

---

## Author

**Samprati Gaurav**
B.Tech CSE (Cyber Security) · Dayananda Sagar University

[![GitHub](https://img.shields.io/badge/GitHub-sampratigaurav-181717?logo=github)](https://github.com/sampratigaurav)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Built with PyTorch · FastAPI · Next.js · Modal · Vercel*

</div>
