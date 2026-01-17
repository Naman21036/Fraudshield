from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import sys
import pickle 
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    trained_models = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_test_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_test_prob = model.decision_function(X_test)

            score = roc_auc_score(y_test, y_test_prob)

            report[name] = score
            trained_models[name] = model

        except Exception as e:
            report[name] = -1
            print(f"Model {name} failed: {e}")

    return report, trained_models

def load_object(file_path):
    try:
        if not os.path.isabs(file_path):
            BASE_DIR = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            file_path = os.path.join(BASE_DIR, file_path)

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
