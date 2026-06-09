# FraudShield 🛡️

### Real Time Transaction Fraud Detection System

FraudShield is an end to end machine learning powered fraud detection platform designed to identify suspicious financial transactions in real time.

Built with FastAPI and modern machine learning techniques, the system predicts fraud probability, generates risk scores, and provides fast API responses suitable for production environments.

### 🌐 Live Demo

**API Endpoint:** https://fraudshield-project.onrender.com/

**GitHub Repository:** https://github.com/Naman21036/FraudShield

---

## Overview

Financial fraud detection is one of the most challenging machine learning applications due to:

* Extreme class imbalance
* Rapidly evolving fraud patterns
* High cost of false negatives
* Real time decision requirements

FraudShield addresses these challenges through a complete machine learning pipeline that combines data preprocessing, feature scaling, model training, and API based deployment.

The system predicts the probability of fraud instead of making only binary decisions, enabling risk based transaction monitoring.

---

## Features

### Fraud Prediction API

* Real time transaction scoring
* REST API powered by FastAPI
* Millisecond level prediction latency

### Machine Learning Pipeline

* Automated preprocessing
* Feature scaling
* Model serialization
* Reproducible training workflow

### Imbalanced Data Handling

* SMOTE based oversampling
* Class weighting techniques
* Threshold optimization

### Risk Scoring

* Fraud probability output
* Confidence based classification
* Explainable risk assessment

### Production Ready Deployment

* FastAPI backend
* Docker compatible architecture
* Cloud deployment support

---

## System Architecture

```text
Transaction Input
        │
        ▼
Input Validation
(FastAPI + Pydantic)
        │
        ▼
Feature Preprocessing
        │
        ▼
Feature Scaling
        │
        ▼
Trained ML Model
        │
        ▼
Fraud Probability
        │
        ▼
Risk Classification
        │
        ▼
JSON Response
```

---

## Tech Stack

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* Scikit Learn
* Imbalanced Learn

### Data Processing

* Pandas
* NumPy

### Model Experimentation

* Random Forest
* XGBoost
* LightGBM

### Deployment

* Render
* Docker Ready

---

## API Usage

### Endpoint

```http
POST /predict
```

### Request Example

```json
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
```

### Response Example

```json
{
  "fraud_prediction": 1,
  "fraud_label": "Fraud",
  "fraud_probability": 0.91
}
```

---

## Dataset

The project is trained on publicly available financial fraud datasets.

### Recommended Sources

* Kaggle Credit Card Fraud Detection Dataset
* PaySim Mobile Money Fraud Dataset

To preserve privacy, transaction features are anonymized using PCA transformed variables.

---

## Model Evaluation

Since fraud detection is an imbalanced classification problem, traditional accuracy metrics are insufficient.

Evaluation focuses on:

* Precision
* Recall
* F1 Score
* ROC AUC
* Precision Recall Tradeoff

### Why Not Accuracy?

A model predicting every transaction as legitimate can achieve over 99% accuracy while completely failing to detect fraud.

For this reason, FraudShield prioritizes Recall and F1 Score.

---

## Project Structure

```text
FraudShield/
│
├── dev/
│   └── backend/
│       ├── app.py
│       ├── requirements.txt
│
├── artifacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── model.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── utils/
│
└── README.md
```

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/Naman21036/FraudShield.git
cd FraudShield/dev/backend
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn app:app --reload --port 10000
```

### Open Interactive API Documentation

```text
http://localhost:10000/docs
```

---

## Deployment

FraudShield is designed for deployment on:

* Render
* Railway
* Fly.io
* Docker Containers

Production start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## Future Enhancements

* Kafka Based Streaming Fraud Detection
* Model Drift Monitoring
* Automated Retraining Pipelines
* Feature Store Integration
* Explainable AI Dashboard
* Hybrid Rule Engine + ML Detection
* Real Time Alerting System

---

## Contributors

### Naman Gupta
---

## License

This project is released under the MIT License.

---

⭐ If you found this project useful, consider starring the repository.
