import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load("rf.pkl")

st.title("Liver Disease Prediction")

st.write("Enter patient details to check if they have liver disease.")

# Example input fields
age = st.number_input("Age of the patient", min_value=1, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, step=0.1)
direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, step=0.1)
alk_phos = st.number_input("Alkphos Alkaline Phosphotase", min_value=0.0, step=0.1)
sgpt = st.number_input("Sgpt Alamine Aminotransferase", min_value=0.0, step=0.1)
sgot = st.number_input("Sgot Aspartate Aminotransferase", min_value=0.0, step=0.1)
total_proteins = st.number_input("Total Proteins", min_value=0.0, step=0.1)
albumin = st.number_input("ALB Albumin", min_value=0.0, step=0.1)
ag_ratio = st.number_input("A/G Ratio Albumin and Globulin Ratio", min_value=0.0, step=0.1)

# Convert gender to numeric (if needed)
gender_num = 1 if gender == "Male" else 0

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, gender_num, total_bilirubin, direct_bilirubin, alk_phos, sgpt, sgot, total_proteins, albumin, ag_ratio]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("The patient is likely to have liver disease.")
    else:
        st.success("The patient is unlikely to have liver disease.")
