import streamlit as st
import requests


# Page Config

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)


# Custom Styling

st.markdown("""
<style>
.main {
    background-color: #fffdf5;
}

.title {
    text-align: center;
    color: #d4a017;
    font-size: 34px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}

div.stButton > button {
    background-color: #f4c430;
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    border: none;
    transition: 0.3s ease;
    box-shadow: 0 0 10px #f4c430;
}

div.stButton > button:hover {
    background-color: white;
    color: #d4a017;
    border: 2px solid #f4c430;
    box-shadow: 0 0 20px #f4c430, 0 0 40px #f4c430;
    transform: scale(1.02);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #fff8dc;
    border: 1px solid #f4c430;
    margin-top: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# Header

st.markdown('<div class="title">Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a telecom customer is likely to churn</div>', unsafe_allow_html=True)


# Inputs

tenure = st.number_input("Tenure", min_value=0.0, value=12.0)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.5)
total = st.number_input("Total Charges", min_value=0.0, value=850.0)

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])


# Predict Button

if st.button("Predict Churn"):

    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "InternetService": internet
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            probability = result["probability_churn"]

            label = "Likely to Churn ⚠️" if prediction == 1 else "Likely to Stay "

            st.markdown(f"""
            <div class="result-box">
                <h3>{label}</h3>
                <p><strong>Churn Probability:</strong> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("API request failed.")

    except Exception as e:
        st.error(f"Connection error: {e}")