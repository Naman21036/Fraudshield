import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

_model = None
_scaler = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _scaler
    try:
        logger.info("Loading artifacts from %s", ARTIFACTS)
        with open(ARTIFACTS / "best_model.pkl", "rb") as f:
            _model = pickle.load(f)
        with open(ARTIFACTS / "scaler.pkl", "rb") as f:
            _scaler = pickle.load(f)
        logger.info("Loaded model: %s", type(_model).__name__)
    except FileNotFoundError as exc:
        logger.critical("Artifact file not found: %s", exc)
        raise
    except Exception as exc:
        logger.critical("Failed to load artifacts: %s", exc)
        raise
    yield
    _model = _scaler = None


app = FastAPI(
    title="FraudShield",
    description="Real-time credit card fraud detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionInput(BaseModel):
    Time: Annotated[float, Field(gt=0, description="Seconds elapsed since first transaction in dataset")]
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: Annotated[float, Field(gt=0, description="Transaction amount")]


class PredictionResponse(BaseModel):
    fraud_prediction: int
    fraud_label: str
    fraud_probability: float | None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/", include_in_schema=False)
def root():
    return {"name": "FraudShield API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(data: TransactionInput):
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = pd.DataFrame([data.model_dump()])[FEATURE_COLS]

    try:
        scaled = _scaler.transform(df)
        prediction = int(_model.predict(scaled)[0])
        probability = (
            float(_model.predict_proba(scaled)[0][1])
            if hasattr(_model, "predict_proba")
            else None
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        fraud_prediction=prediction,
        fraud_label="Fraud" if prediction == 1 else "Legitimate",
        fraud_probability=probability,
    )
