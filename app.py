import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('heart_disease_model.pkl')

st.title("Heart Disease Prediction")

# User input
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Serum Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", ["normal", "ST-T abnormality", "left ventricular hypertrophy"])
thalch = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise")
slope = st.selectbox("Slope of ST segment", ["upsloping", "flat", "downsloping"])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", 0, 4)
thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

# Map categorical inputs
mapping = {"Male":1, "Female":0, "Yes":1, "No":0,
           "typical angina":0, "atypical angina":1, "non-anginal pain":2, "asymptomatic":3,
           "normal":0, "ST-T abnormality":1, "left ventricular hypertrophy":2,
           "upsloping":0, "flat":1, "downsloping":2,
           "fixed defect":1, "reversible defect":2}

input_data = pd.DataFrame({
    "age":[age],
    "sex":[mapping[sex]],
    "cp":[mapping[cp]],
    "trestbps":[trestbps],
    "chol":[chol],
    "fbs":[mapping[fbs]],
    "restecg":[mapping[restecg]],
    "thalch":[thalch],
    "exang":[mapping[exang]],
    "oldpeak":[oldpeak],
    "slope":[mapping[slope]],
    "ca":[ca],
    "thal":[mapping[thal]],
    "dataset":[0]  # default
})

if st.button("Predict"):
    result = model.predict(input_data)
    if result[0]==0:
        st.success("No Heart Disease")
    else:
        st.error("Heart Disease Detected")
