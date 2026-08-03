# Adult Income Prediction — ML & DL Project

Real-world industry-style end-to-end project predicting whether a person's annual
income exceeds $50K, using the UCI Adult / Census Income dataset (45,174 real,
cleaned records).

## Folder Structure
```
adult-income-app/
├── app.py                                  <- Streamlit live prediction app
├── requirements.txt                        <- Python dependencies
├── render.yaml                             <- Render.com deploy config
├── runtime.txt                             <- Pins Python version (3.11.9)
├── README.md
├── Adult_Income_Prediction_ML_DL.ipynb     <- Full notebook: EDA, ML, DL, training
├── adult_income_full.csv                   <- Cleaned dataset (45,174 rows)
├── outputs/                                <- Saved chart images from EDA
└── models/                                 <- Saved production model artifacts
    ├── best_ml_model.pkl
    ├── dl_model.keras
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── target_encoder.pkl
    ├── feature_names.pkl
    └── results_summary.json
```

## Deploy on Render.com
1. Push this entire folder to a GitHub repo (root of repo = contents of this folder)
2. Go to render.com -> New Web Service -> connect this repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy

## View the analysis
Open `Adult_Income_Prediction_ML_DL.ipynb` directly on GitHub (renders automatically)
to see the full EDA, model training, graphs, and evaluation — no need to run it locally.

## Results Summary (Test Set)

| Model                | Accuracy | Precision | Recall | F1    | ROC-AUC |
|-----------------------|----------|-----------|--------|-------|---------|
| Logistic Regression   | 0.822    | 0.721     | 0.460  | 0.561 | 0.844   |
| Random Forest          | 0.858    | 0.790     | 0.583  | 0.671 | 0.915   |
| **XGBoost (Best ML)**  | **0.864**| 0.770     | 0.647  | 0.703 | **0.925**|
| Deep Neural Network    | 0.841    | 0.721     | 0.587  | 0.647 | 0.901   |
