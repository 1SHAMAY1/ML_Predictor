import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Medical Insurance Predictor", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'insurance_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')

st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #10b981;
        color: white;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Medical Insurance Predictor")
st.write("This model is trained on real hospital data.")

st.sidebar.title("System Logs")
if os.path.exists(MODEL_PATH):
    st.sidebar.success(f"Model found at: {MODEL_PATH}")
else:
    st.sidebar.error(f"Model MISSING at: {MODEL_PATH}")

if os.path.exists(ENCODER_PATH):
    st.sidebar.success(f"Encoders found at: {ENCODER_PATH}")
else:
    st.sidebar.error(f"Encoders MISSING at: {ENCODER_PATH}")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.error("Critical Error: Model or Encoders not found on the system!")
    st.info("Try running 'python train_model.py' inside the ML_Predictor folder.")
else:
    with st.spinner("Loading prediction model..."):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        sex = st.selectbox("Gender", options=list(encoders['sex'].classes_))
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    
    with col2:
        children = st.selectbox("Children", options=[0, 1, 2, 3, 4, 5])
        smoker = st.selectbox("Smoker?", options=list(encoders['smoker'].classes_))
        region = st.selectbox("Region", options=list(encoders['region'].classes_))

    if st.button("Calculate Estimated Cost"):
        sex_enc = encoders['sex'].transform([sex])[0]
        smoker_enc = encoders['smoker'].transform([smoker])[0]
        region_enc = encoders['region'].transform([region])[0]
        
        input_df = pd.DataFrame([[age, sex_enc, bmi, children, smoker_enc, region_enc]], 
                               columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        
        prediction = model.predict(input_df)[0]
        
        st.success(f"Estimated Annual Premium: ${prediction:,.2f}")
        st.sidebar.write(f"Last Prediction: ${prediction:,.2f}")

st.markdown("---")
st.caption("Trained on the Kaggle Insurance Dataset.")
