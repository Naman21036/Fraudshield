Transaction Fraud Detection System

An end to end machine learning based fraud detection system for financial transactions built using FastAPI. The project focuses on real time risk scoring behavior based features and production ready ML deployment.

Project Overview

Financial fraud detection is a highly imbalanced and time sensitive problem. This project detects fraudulent transactions by analyzing transaction attributes and user behavior patterns.

The system exposes trained ML models through FastAPI endpoints to provide fast and explainable fraud predictions.

Key Features

Real time fraud prediction API

Machine learning based risk scoring

Handles highly imbalanced transaction data

Behavior based feature engineering

Explainable predictions using feature importance

Modular and deployment ready architecture

Tech Stack

Python

Scikit learn

FastAPI

Pandas NumPy

Uvicorn

Docker optional

Dataset

The model is trained on publicly available credit card transaction datasets with fraud labels. You can use datasets from Kaggle such as Credit Card Fraud Detection or PaySim.

System Architecture

Client sends transaction details

FastAPI validates request

Feature engineering pipeline processes input

ML model predicts fraud probability

Risk decision returned in real time

API Endpoints POST /predict

Request body example

{ "user_id": "U123", "amount": 18500, "merchant": "electronics", "timestamp": "2026-01-14T12:30:00", "location": "Delhi", "device": "mobile" }

Response example

{ "fraud_probability": 0.91, "prediction": "fraud", "risk_level": "high" }

Project Structure ├── app │ ├── main.py │ ├── models │ ├── schemas │ ├── services │ └── utils ├── data ├── notebooks ├── training ├── requirements.txt └── README.md

Model Evaluation

Evaluation focuses on

Precision

Recall

F1 Score

PR AUC

Accuracy is avoided due to class imbalance.

How to Run Locally

Clone the repository

Install dependencies

Start the FastAPI server

pip install -r requirements.txt uvicorn app.main:app --reload

Future Improvements

Streaming support using Kafka

Automated model retraining

Model drift detection

Rule based and ML hybrid system

Monitoring dashboard

Author

Naman Gupta