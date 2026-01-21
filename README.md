Transaction Fraud Detection System

An end to end machine learning based transaction fraud detection system built using FastAPI.
The project focuses on real time risk scoring, handling highly imbalanced data, and production ready ML deployment.

Project Overview

Financial fraud detection is a highly imbalanced and time sensitive classification problem.
This system identifies fraudulent transactions by analyzing transaction level features and learned behavioral patterns.

A trained machine learning model is exposed via FastAPI endpoints to deliver fast, reliable, and explainable fraud predictions suitable for real world deployment.

Key Features

Real time fraud prediction API

Probabilistic risk scoring instead of binary decisions

Handles extreme class imbalance effectively

Robust preprocessing and feature scaling pipeline

Modular ML training and evaluation workflow

Production ready FastAPI backend

Deployment ready for cloud platforms (Render, Railway, etc.)

Tech Stack

Python

Scikit learn

Pandas, NumPy

FastAPI

Uvicorn

Imbalanced learn

XGBoost, LightGBM (model experimentation)

Dataset

The model is trained using publicly available credit card transaction datasets with fraud labels.

Recommended datasets:

Kaggle Credit Card Fraud Detection Dataset

PaySim Mobile Money Fraud Dataset

Note: Due to privacy concerns, original features are anonymized using PCA based transformations.

System Architecture

Client sends transaction data to the API

FastAPI validates and parses the request

Preprocessing and scaling pipeline transforms the input

Trained ML model predicts fraud probability

Risk score and decision are returned in real time

API Endpoint
POST /predict
Request Body Example
{
  "Time": 12345,
  "V1": -1.23,
  "V2": 0.45,
  "V3": -2.31,
  "V4": 1.12,
  "V5": -0.98,
  "V6": 0.33,
  "V7": -1.87,
  "V8": 0.22,
  "V9": -0.45,
  "V10": -2.01,
  "V11": 1.45,
  "V12": -3.12,
  "V13": 0.56,
  "V14": -4.23,
  "V15": 0.12,
  "V16": -1.45,
  "V17": -2.78,
  "V18": 0.89,
  "V19": -0.33,
  "V20": 0.41,
  "V21": 0.18,
  "V22": -0.27,
  "V23": 0.05,
  "V24": 0.62,
  "V25": -0.14,
  "V26": 0.03,
  "V27": -0.02,
  "V28": 0.01,
  "Amount": 18500
}

Response Example
{
  "fraud_prediction": 1,
  "fraud_label": "Fraud",
  "fraud_probability": 0.91
}

Project Structure
FraudShield/
├── dev/
│   └── backend/
│       ├── app.py
│       ├── requirements.txt
├── artifacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
├── notebooks/
│   ├── eda.ipynb
│   ├── model.ipynb
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── utils/
├── README.md

Model Evaluation Strategy

Due to extreme class imbalance, evaluation focuses on:

Precision

Recall

F1 Score

ROC AUC

Accuracy is intentionally avoided as it is misleading for fraud detection problems.

How to Run Locally
1. Clone the repository
git clone https://github.com/Naman21036/FraudShield.git
cd FraudShield/dev/backend

2. Install dependencies
pip install -r requirements.txt

3. Start FastAPI server
uvicorn app:app --port 10000

4. Open API docs
http://localhost:10000/docs

Deployment

The application is designed for cloud deployment using:

Render

Railway

Fly.io

Start command for production:

uvicorn app:app --host 0.0.0.0 --port $PORT

Future Improvements

Streaming fraud detection using Kafka

Automated model retraining pipelines

Model drift detection

Hybrid rule based + ML decision engine

Monitoring and alerting dashboard

Author

Naman Gupta
Machine Learning and Backend Engineering
Focused on applied ML systems and production deployment