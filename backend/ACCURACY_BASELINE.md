# Distill Accuracy Baseline — April 2026

## Credit Card Fraud Dataset (creditcardfraud, Kaggle)
- Total samples: 284,807
- True anomalies: 492 (0.17% contamination)
- Distill flagged: ~41,857 (14.7%)
- Processing time: 75 seconds
- Known limitation: Autoencoder over-flags on high-variance 
  tabular data with very low contamination rates (<1%)
- Deep SVDD flagged: 35,998 — most conservative model
- Isolation Forest flagged: 9,105 — reasonable range

## MEN_P_dataset (images, custom)
- Total samples: 20
- True anomalies: 3 (15% contamination)  
- Distill flagged with prompt: 2-3 (Fix B zero-shot)
- Distill flagged without prompt: 1-2 (Fix A cosine only)
- Known limitation: small dataset (<100 samples) reduces reliability

## Architecture notes
- Statistical pre-filter: catches data corruption only (10-sigma)
- ML ensemble: 2-of-3 voting reduces false positives
- Fix A: cosine similarity boosts image detection
- Fix B: CLIP zero-shot requires text prompt, images only
