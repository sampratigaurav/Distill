# 🛡️ Distill: Universal Data Sanitization & Poisoning Detection

**Distill** is a high-performance, unsupervised machine learning suite designed to identify and quarantine "poisoned" or anomalous data within datasets. It bridges the gap between raw data collection and model training by providing a deterministic, multi-model defense layer.

Whether you are uploading a CSV of network traffic or a ZIP file of images, Distill translates the data into a universal latent space and mathematically hunts down the outliers.

---

## 🚀 Key Features

* **Universal Feature Extraction:** Automatically routes Tabular (CSV) data through pandas and Image archives (ZIP) through a headless **ResNet-18** model to map all data into a uniform mathematical latent space.
* **Ensemble Defense (Strict Democracy):** Utilizes a rigid 2-out-of-3 consensus voting system between:
  * Dynamic Autoencoders (Reconstruction Error)
  * Deep SVDD (Distance to Semantic Center)
  * Isolation Forests (Decision Function Margins)
* **Explainable AI (XAI):** Moves beyond the "black box" by visualizing **Latent Space Divergence**. For images, it charts the exact ResNet neural feature indices that alienated the image from the clean dataset.
* **Closed-Loop Sanitization:** Provides a one-click "Download Sanitized Dataset" feature that mathematically removes flagged items and returns a pure ZIP/CSV for downstream model training.
* **Real-Time Asynchronous Processing:** Uses **WebSockets** and FastAPI `BackgroundTasks` to stream live PyTorch training progress to the frontend without freezing the browser.
* **Production-Hardened Security:** Implements strict 1GB payload limits at both the Next.js and FastAPI layers to prevent memory-exhaustion DoS attacks.

---

## 🧠 The Architecture & Mathematics

Unlike standard anomaly detectors that rely on simple averages (which are easily skewed by the poison itself), Distill is engineered for **Scale-Invariant Determinism**.

1. **Robust Scaling (MAD/IQR):** We use Median Absolute Deviation (MAD) instead of Mean/Standard Deviation. This ensures the baseline of "normal" is highly resistant to heavy contamination.
2. **The "Zero-Variance" Epsilon Floor:** To prevent "Threshold Squeezing" (where a perfectly clean dataset creates a variance of zero, causing microscopic artifacts to trigger false positives), the scaling engine injects a strict `1e-5` mathematical floor.
3. **Dynamic Thresholding:** The Modified Z-Score boundary shifts dynamically based on dataset size (e.g., stricter thresholds for `n < 100` to prevent micro-batch volatility).

---

## 💻 Tech Stack

**Backend (The ML Engine):**
* **Framework:** FastAPI, Python 3.9
* **Deep Learning:** PyTorch, Torchvision (ResNet-18 weights)
* **Machine Learning:** Scikit-Learn (Isolation Forest, MinMaxScaler)
* **Data Processing:** Pandas, NumPy, OpenCV, Pillow

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
```
2. Start the Backend (FastAPI + PyTorch)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
3. Start the Frontend (Next.js)
```bash
cd ../frontend
npm install
npm run dev
```
Open http://localhost:3000 in your browser.

👨‍💻 Author
Samprati Gaurav
