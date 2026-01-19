from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field,computed_field
import uvicorn
from typing import List, Optional,Literal,Annotated
import pandas as pd
import numpy as np
import pickle
import os

app= FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to FraudShield API"}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "best_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "artifacts", "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

class TransactionData(BaseModel):
    Time: float = Annotated[float,Field(..., description="Number of seconds elapsed between this transaction and the first transaction in the dataset",gt=0)]
    V1: float = Annotated[float,Field(..., description="V1 feature(result of a PCA Dimensionality reduction to protect user identities and sensitive features)")]
    V2: float = Annotated[float,Field(..., description="V2 feature")]
    V3: float = Annotated[float,Field(..., description="V3 feature")]
    V4: float = Annotated[float,Field(..., description="V4 feature")]
    V5: float = Annotated[float,Field(..., description="V5 feature")]
    V6: float = Annotated[float,Field(..., description="V6 feature")]
    V7: float = Annotated[float,Field(..., description="V7 feature")]
    V8: float = Annotated[float,Field(..., description="V8 feature")]
    V9: float = Annotated[float,Field(..., description="V9 feature")]
    V10: float = Annotated[float,Field(..., description="V10 feature")]
    V11: float = Annotated[float,Field(..., description="V11 feature")]
    V12: float = Annotated[float,Field(..., description="V12 feature")]
    V13: float = Annotated[float,Field(..., description="V13 feature")]
    V14: float = Annotated[float,Field(..., description="V14 feature")]
    V15: float = Annotated[float,Field(..., description="V15 feature")]
    V16: float = Annotated[float,Field(..., description="V16 feature")]
    V17: float = Annotated[float,Field(..., description="V17 feature")]
    V18: float = Annotated[float,Field(..., description="V18 feature")]
    V19: float = Annotated[float,Field(..., description="V19 feature")]
    V20: float = Annotated[float,Field(..., description="V20 feature")]
    V21: float = Annotated[float,Field(..., description="V21 feature")]
    V22: float = Annotated[float,Field(..., description="V22 feature")]
    V23: float = Annotated[float,Field(..., description="V23 feature")]
    V24: float = Annotated[float,Field(..., description="V24 feature")]
    V25: float = Annotated[float,Field(..., description="V25 feature")]
    V26: float = Annotated[float,Field(..., description="V26 feature")]
    V27: float = Annotated[float,Field(..., description="V27 feature")]
    V28: float = Annotated[float,Field(..., description="V28 feature")]
    Amount: float = Annotated[float,Field(..., description="Amount of the transaction",gt=0)]


@app.post('/predict')
def predict_fraud(data: TransactionData):
    
    try:
        input_df= pd.DataFrame([{
            'Time': data.Time,
            'V1': data.V1,
            'V2': data.V2,
            'V3': data.V3,
            'V4': data.V4,
            'V5': data.V5,
            'V6': data.V6,
            'V7': data.V7,
            'V8': data.V8,
            'V9': data.V9,
            'V10': data.V10,
            'V11': data.V11,
            'V12': data.V12,
            'V13': data.V13,
            'V14': data.V14,
            'V15': data.V15,
            'V16': data.V16,
            'V17': data.V17,
            'V18': data.V18,
            'V19': data.V19,
            'V20': data.V20,
            'V21': data.V21,
            'V22': data.V22,
            'V23': data.V23,
            'V24': data.V24,
            'V25': data.V25,
            'V26': data.V26,
            'V27': data.V27,
            'V28': data.V28,
            'Amount': data.Amount
        }])
        input_arr = scaler.transform(input_df)
        prediction = int(model.predict(input_arr)[0])
        probability = (
            float(model.predict_proba(input_arr)[0][1])
            if hasattr(model, "predict_proba")
            else None
        )

        response = {
            "fraud_prediction": prediction,
            "fraud_label": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": probability
        }

        return JSONResponse(
            status_code=200,
            content=response
        )
    

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "detail": str(e)
            }
        )
    