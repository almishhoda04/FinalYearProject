import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Load Models
# -------------------------------

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "Models")

rf = joblib.load(os.path.join(MODEL_DIR, "rf_base.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_base.pkl"))
mlp = joblib.load(os.path.join(MODEL_DIR, "mlp_base.pkl"))
meta = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Liver Disease Prediction using SCEM")

st.write(
    "Enter the patient's clinical details below to predict the likelihood of liver disease."
)

# -------------------------------
# Input Fields
# -------------------------------

age = st.number_input("Age", min_value=1, max_value=100)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

total_bilirubin = st.number_input(
    "Total Bilirubin",
    min_value=0.0,
    step=0.1
)

direct_bilirubin = st.number_input(
    "Direct Bilirubin",
    min_value=0.0,
    step=0.1
)

alk = st.number_input(
    "Alkaline Phosphotase",
    min_value=0.0,
    step=0.1
)

sgpt = st.number_input(
    "SGPT",
    min_value=0.0,
    step=0.1
)

sgot = st.number_input(
    "SGOT",
    min_value=0.0,
    step=0.1
)

protein = st.number_input(
    "Total Proteins",
    min_value=0.0,
    step=0.1
)

albumin = st.number_input(
    "Albumin",
    min_value=0.0,
    step=0.1
)

ag = st.number_input(
    "Albumin / Globulin Ratio",
    min_value=0.0,
    step=0.1
)

gender_num = 1 if gender == "Male" else 0

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([[
        age,
        total_bilirubin,
        direct_bilirubin,
        alk,
        sgpt,
        sgot,
        protein,
        albumin,
        ag,
        gender_num
    ]],
    columns=[
        "Age of the patient",
        "Total Bilirubin",
        "Direct Bilirubin",
        "Alkphos Alkaline Phosphotase",
        "Sgpt Alanine Aminotransferase",
        "Sgot Aspartate Aminotransferase",
        "Total Protiens",
        "ALB Albumin",
        "A/G Ratio Albumin and Globulin Ratio",
        "Gender"
    ])

    # Base Model Predictions
    rf_prob = rf.predict_proba(input_df)[:, 1]
    xgb_prob = xgb.predict_proba(input_df)[:, 1]
    mlp_prob = mlp.predict_proba(input_df)[:, 1]

    meta_input = pd.DataFrame({
        "RF": rf_prob,
        "XGB": xgb_prob,
        "MLP": mlp_prob
    })

    final_prob = meta.predict_proba(meta_input)[0][1]
    final_pred = meta.predict(meta_input)[0]

    disease_prob = final_prob * 100
    healthy_prob = (1 - final_prob) * 100

    if final_pred == 1:
        st.error("⚠ Liver Disease Detected")
    else:
        st.success("✅ No Liver Disease Detected")

    st.write(f"**Probability of Liver Disease:** {disease_prob:.2f}%")
    st.write(f"**Probability of Healthy:** {healthy_prob:.2f}%")