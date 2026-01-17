import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils.metrics import save_object

@dataclass
class ScalerConfig:
    scaler_object_path: str = os.path.join("artifacts", "scaler.pkl")
class Scaler:
    def __init__(self):
        self.scaler_config = ScalerConfig()

    def get_scaler_object(self, X: pd.DataFrame):
        try:
            numeric_scaled_cols = ["Time", "Amount"]
            pca_cols = [col for col in X.columns if col not in numeric_scaled_cols]

            logging.info(f"Scaling columns: {numeric_scaled_cols}")
            logging.info(f"Passing through PCA columns: {pca_cols}")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_scaled_cols),
                    ("pca", "passthrough", pca_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train columns: {train_df.columns.tolist()}")
            logging.info("Train and test data loaded")

            target_column = "Class"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessing_obj = self.get_scaler_object(X_train)

            logging.info("Fitting preprocessing object on training data")

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path=self.scaler_config.scaler_object_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved")

            return (
                train_arr,
                test_arr,
                self.scaler_config.scaler_object_path
            )

        except Exception as e:
            raise CustomException(e, sys)
