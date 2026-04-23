from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd


# Load model

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_churn_model.pkl"

model = joblib.load(MODEL_PATH)


expected_features = model.named_steps["preprocessor"].feature_names_in_


# App

app = FastAPI(title="Customer Churn Prediction API")

# 
# Input Schema

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    InternetService: str


# Root

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}


# Predict

@app.post("/predict")
def predict(data: CustomerData):

    try:
        # convert input to dataframe
        input_df = pd.DataFrame([data.dict()])

        # ensure all expected features are present 
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0

        
        input_df = input_df[expected_features]

        # prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "probability_churn": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}