# Adult Income Prediction — ML (Streamlit Cloud Ready)

Real-world project predicting whether a person's annual income exceeds $50K,
using the UCI Adult / Census Income dataset (45,174 real, cleaned records).

This version's `app.py` is **ML-only** (XGBoost) — no TensorFlow dependency —
so it deploys cleanly and quickly on Streamlit Community Cloud with zero errors.
The Deep Learning model is still trained, evaluated, and saved
(`models/dl_model.keras`) inside the notebook for reference/comparison.

## Folder Structure
```
adult-income-streamlit/
├── app.py                                  <- Streamlit app (ML-only, lightweight)
├── requirements.txt                        <- No TensorFlow — fast, error-free installs
├── runtime.txt                             <- Pins Python version (3.11.9)
├── README.md
├── Adult_Income_Prediction_ML_DL.ipynb     <- Full notebook: EDA, ML AND DL training/comparison
├── adult_income_full.csv                   <- Cleaned dataset (45,174 rows)
├── outputs/                                <- Saved chart images from EDA
└── models/
    ├── best_ml_model.pkl                   <- Used by app.py
    ├── scaler.pkl                          <- Used by app.py
    ├── label_encoders.pkl                  <- Used by app.py
    ├── target_encoder.pkl                  <- Used by app.py
    ├── feature_names.pkl                   <- Used by app.py
    ├── results_summary.json                <- Used by app.py
    └── dl_model.keras                      <- Saved DL model (from notebook, not used by app.py)
```

## Deploy on Streamlit Community Cloud
1. Push this entire folder's contents to a GitHub repo (root of repo = contents of this folder)
2. Go to https://share.streamlit.io -> sign in with GitHub
3. Click "New app" -> select this repo -> Main file path: `app.py`
4. Click "Deploy" — done in ~2 minutes, no manual build/install commands needed

## Results Summary (Test Set)

| Model                | Accuracy | Precision | Recall | F1    | ROC-AUC |
|-----------------------|----------|-----------|--------|-------|---------|
| Logistic Regression   | 0.822    | 0.721     | 0.460  | 0.561 | 0.844   |
| Random Forest          | 0.858    | 0.790     | 0.583  | 0.671 | 0.915   |
| **XGBoost (used by app)** | **0.864** | 0.770 | 0.647  | 0.703 | **0.925**|
| Deep Neural Network (notebook only) | 0.841 | 0.721 | 0.587 | 0.647 | 0.901 |
