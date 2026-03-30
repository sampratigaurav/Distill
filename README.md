# 🛡️ Distill: Universal Data Sanitization & Poisoning Detection

**Distill** is a high-performance, unsupervised machine learning suite designed to identify and quarantine "poisoned" or anomalous data within massive datasets. It bridges the gap between raw data collection and model training by providing a deterministic, multi-model defense layer.

Whether you are uploading a CSV of network logs, a ZIP file of images, or a dataset full of free-text customer reviews, Distill translates the data into a universal latent space and mathematically hunts down the outliers.

---

## 🚀 Enterprise-Grade Features

* **Universal Feature Extraction (Numbers, Images, & Text):** * Automatically routes numeric/categorical data through pandas scaling/encoding.
  * Routes Images through a headless **ResNet-18** model.
  * Routes Free-Text through a local **SentenceTransformer (all-MiniLM-L6-v2)** to detect semantic anomalies.
* **Smart Chunking (Streaming Architecture):** Built to handle datasets of theoretically infinite size. Distill streams data in memory-safe chunks, preventing Out-Of-Memory (OOM) crashes even when processing gigabytes of data.
* **Ensemble Defense (Strict Democracy):** Utilizes a rigid 2-out-of-3 consensus voting system between:
  * Dynamic Autoencoders (Reconstruction Error)
  * Deep SVDD (Distance to Semantic Center)
  * Isolation Forests (Decision Function Margins)
* **Explainable AI (XAI):** Moves beyond the "black box." Distill charts Latent Space Divergence for images, highlights absolute reconstruction errors for tabular data, and extracts the exact text snippets causing high semantic deviation.
* **Automated PDF Sanitization Receipts:** Upon cleaning a dataset, the server generates a professional PDF report detailing dataset health, anomaly percentages, and the top 5 most toxic data points removed, packaged neatly with your clean data.

---

## 🔒 Production-Hardened Security

Unlike standard anomaly detectors, Distill is engineered for scale-invariant determinism and adversarial defense:
1. **Real-Time Payload Streaming:** Enforces a strict 1GB payload limit by physically counting bytes as they stream into the ASGI layer, defeating HTTP Header spoofing DoS attacks.
2. **Zip Bomb Defusal:** Unpacks archives via an x-ray streaming tunnel that halts extraction the exact millisecond uncompressed data exceeds safety limits, ignoring forged metadata.
3. **Zero-RAM Downloads:** Utilizes a "Two-Step Pickup Box" architecture. Cleaned datasets are fetched via native browser download managers, completely bypassing JavaScript `Blob` memory traps to prevent browser tab crashes.
4. **Robust Scaling (MAD/IQR):** Uses Median Absolute Deviation with a strict `1e-5` variance floor to prevent clean datasets from triggering microscopic false positives.

---

## 💻 Tech Stack

**Backend (The ML Engine):**
* **Framework:** FastAPI, Python 3.9
* **Deep Learning:** PyTorch, Torchvision (ResNet-18)
* **NLP:** Sentence-Transformers (all-MiniLM-L6-v2)
* **Machine Learning:** Scikit-Learn (Isolation Forest), Pandas, NumPy
* **Reporting:** FPDF2

**Frontend (The Dashboard):**
* **Framework:** Next.js 14 (App Router), React
* **Styling:** Tailwind CSS
* **Data Visualization:** Recharts
* **Icons:** Lucide-React

---

## ⚙️ Running Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/sampratigaurav/Distill.git](https://github.com/sampratigaurav/Distill.git)
cd Distill
2. Start the Backend (FastAPI + PyTorch)
Bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
3. Start the Frontend (Next.js)
Bash
cd ../frontend
npm install
npm run dev
Open http://localhost:3000 in your browser.

👨‍💻 Author: Samprati Gaurav