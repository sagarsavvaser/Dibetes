import streamlit as st
import pickle
import numpy as np
import os

# Make paths work on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
with open(os.path.join(BASE_DIR, "logistic_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# App title
st.title("Diabetes Prediction App")

# Sidebar inputs
st.sidebar.header("Patient Data")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:
        st.success("The patient is unlikely to have diabetes.")
