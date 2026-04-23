# Customer Churn Prediction API

## Overview

This project is an end-to-end **tabular machine learning product** that
predicts whether a telecom customer is likely to churn.

The solution includes: - Data cleaning and exploratory analysis -
Feature preprocessing pipeline - Logistic Regression model -
Hyperparameter tuning with GridSearchCV - Model explainability with
SHAP - FastAPI inference API - Swagger interactive testing - Serialized
model for deployment

------------------------------------------------------------------------

## Project Structure

``` bash
project/
│── api/
│   └── main.py
│── data/
│   ├── processed/
│           └── clean_data.csv 
│   └── raw/
│          └── customer_churn.csv  
│── models/
│   └── logistic_churn_model.pkl
│── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_model_training.ipynb
│── reports/
│── .dockerignore
│── .gitignore
│── Dockerfile
│── requirements.txt
│── README.md
```

------------------------------------------------------------------------

## Model Training

Pipeline includes:

-   `StandardScaler()` for numeric features
-   `OneHotEncoder()` for categorical features
-   `ColumnTransformer()`
-   `LogisticRegression(class_weight="balanced")`

Hyperparameter tuning:

``` python
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["lbfgs", "liblinear"],
    "classifier__penalty": ["l2"]
}
```

Evaluation: - Accuracy ≈ 0.74 - Recall for churn ≈ 0.79 - Mean CV F1 ≈
0.63

------------------------------------------------------------------------

## Running the API

Start server:

``` bash
uvicorn api.main:app --reload
```

Open Swagger:

``` bash
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

## Prediction Endpoint

### POST `/predict`

Example JSON request:

``` json
{
  "tenure": 12,
  "MonthlyCharges": 70,
  "TotalCharges": 800,
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "InternetService": "Fiber optic"
}
```

Example response:

``` json
{
  "prediction": 1,
  "probability_churn": 0.78
}
```

Where: - `prediction = 1` → customer likely to churn - `prediction = 0`
→ customer likely to stay

------------------------------------------------------------------------

## Testing in Swagger UI

1.  Open `/docs`
2.  Expand **POST /predict**
3.  Click **Try it out**
4.  Paste the JSON body
5.  Click **Execute**
6.  View prediction response

------------------------------------------------------------------------

## Saved Model

Model file:

``` bash
models/logistic_churn_model.pkl
```

The saved pipeline contains: - preprocessing - encoding - scaling -
classifier

------------------------------------------------------------------------

## Tools Used

-   Python
-   pandas
-   scikit-learn
-   FastAPI
-   SHAP
-   joblib

------------------------------------------------------------------------

## Author

Balqees Adel
