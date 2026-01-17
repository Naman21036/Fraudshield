from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import sys
import pickle 
import os

from src.exception import CustomException
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np


def evaluate_models(X_train,y_train,X_test,y_test,models,param_grid,scoring="roc_auc"):
    report = {}
    trained_models = {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid[name],
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=0
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            if hasattr(best_model, "predict_proba"):
                y_test_prob = best_model.predict_proba(X_test)[:, 1]
            else:
                y_test_prob = best_model.decision_function(X_test)

            score = roc_auc_score(y_test, y_test_prob)

            report[name] = score
            trained_models[name] = best_model

        except Exception as e:
            report[name] = None
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
