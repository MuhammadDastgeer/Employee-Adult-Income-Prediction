"""
Adult Census Income Prediction — Streamlit Production App
Loads the models & preprocessing artifacts saved by the training notebook
and serves live predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Income Prediction | ML & DL",
    page_icon="💰",
    layout="wide"
)

# ----------------------------------------------------------------------------
# Load artifacts (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    ml_model = joblib.load("models/best_ml_model.pkl")
    dl_model = keras.models.load_model("models/dl_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    target_le = joblib.load("models/target_encoder.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    with open("models/results_summary.json") as f:
        results = json.load(f)
    return ml_model, dl_model, scaler, encoders, target_le, feature_names, results

ml_model, dl_model, scaler, encoders, target_le, feature_names, results = load_artifacts()

ml_model_name = type(ml_model).__name__
ml_display_names = {
    "LogisticRegression": "Logistic Regression",
    "RandomForestClassifier": "Random Forest",
    "XGBClassifier": "XGBoost"
}
ml_display_name = ml_display_names.get(ml_model_name, ml_model_name)

# ----------------------------------------------------------------------------
# Sidebar — model info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Model Info")
    st.markdown(f"**Best ML Model:** {ml_display_name}")
    st.markdown("**Deep Learning Model:** Neural Network (Keras)")
    st.divider()

    st.subheader("Test Set Performance")
    results_df = pd.DataFrame(results).T.round(3)
    st.dataframe(results_df, use_container_width=True)

    st.divider()
    st.caption("Dataset: UCI Adult / Census Income (45,174 real records)")
    st.caption("Task: Predict whether income exceeds $50K/yr")

# ----------------------------------------------------------------------------
# Main header
# ----------------------------------------------------------------------------
st.title("💰 Adult Income Prediction")
st.markdown("### Predict whether a person's annual income exceeds **$50,000** using ML & Deep Learning")
st.markdown("Fill in the details below and get an instant, live production prediction.")
st.divider()

# ----------------------------------------------------------------------------
# Options for categorical inputs (taken from training data label encoders)
# ----------------------------------------------------------------------------
def options_for(col):
    return list(encoders[col].classes_)

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal")
    age = st.slider("Age", 17, 90, 35)
    sex = st.selectbox("Sex", options_for("sex"))
    race = st.selectbox("Race", options_for("race"))
    native_country = st.selectbox(
        "Native Country", options_for("native_country"),
        index=options_for("native_country").index("United-States")
        if "United-States" in options_for("native_country") else 0
    )
    relationship = st.selectbox("Relationship", options_for("relationship"))
    marital_status = st.selectbox("Marital Status", options_for("marital_status"))

with col2:
    st.subheader("🎓 Education & Work")
    education = st.selectbox("Education Level", options_for("education"))
    education_num_map = {
        'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
        '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
        'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
        'Prof-school': 15, 'Doctorate': 16
    }
    education_num = education_num_map.get(education, 9)
    st.caption(f"Education-Num (auto): **{education_num}**")

    workclass = st.selectbox("Workclass", options_for("workclass"))
    occupation = st.selectbox("Occupation", options_for("occupation"))
    hours_per_week = st.slider("Hours Worked per Week", 1, 99, 40)

with col3:
    st.subheader("💵 Financial")
    fnlwgt = st.number_input("Final Weight (Census sampling weight)", min_value=10000,
                              max_value=1500000, value=180000, step=1000)
    capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0, step=100)
    capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=5000, value=0, step=50)

    st.markdown("")
    model_choice = st.radio(
        "🤖 Choose Prediction Model",
        [f"Machine Learning ({ml_display_name})", "Deep Learning (Neural Network)"],
        horizontal=False
    )

st.divider()

# ----------------------------------------------------------------------------
# Predict button
# ----------------------------------------------------------------------------
if st.button("🔮 Predict Income Class", type="primary", use_container_width=True):

    sample = {
        'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education,
        'education_num': education_num, 'marital_status': marital_status,
        'occupation': occupation, 'relationship': relationship, 'race': race,
        'sex': sex, 'capital_gain': capital_gain, 'capital_loss': capital_loss,
        'hours_per_week': hours_per_week, 'native_country': native_country
    }

    row = pd.DataFrame([sample])
    for col, le in encoders.items():
        row[col] = le.transform(row[col].astype(str))
    row = row[feature_names]
    row_scaled = scaler.transform(row)

    use_dl = model_choice.startswith("Deep Learning")

    if use_dl:
        prob = float(dl_model.predict(row_scaled, verbose=0)[0][0])
        used_model_name = "Deep Neural Network"
    else:
        prob = ml_model.predict_proba(row_scaled)[0][1]
        used_model_name = ml_display_name

    pred_label = target_le.inverse_transform([int(prob >= 0.5)])[0]

    st.divider()
    res_col1, res_col2 = st.columns([1, 1.3])

    with res_col1:
        if pred_label == ">50K":
            st.success(f"### ✅ Predicted Income: **{pred_label}**")
        else:
            st.info(f"### 📊 Predicted Income: **{pred_label}**")
        st.metric("Model Used", used_model_name)
        st.metric("Probability of >50K", f"{prob*100:.2f}%")

    with res_col2:
        st.markdown("#### Prediction Confidence")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"P(Income > $50K) = {prob:.4f}   |   P(Income <= $50K) = {1-prob:.4f}")

        with st.expander("🔍 See input sent to model"):
            st.json(sample)

st.divider()
st.caption("Built as an end-to-end ML/DL production project — models trained in the accompanying Jupyter notebook.")
