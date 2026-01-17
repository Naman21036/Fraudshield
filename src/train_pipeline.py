import sys
from src.exception import CustomException
from src.logger import logging

from src.preprocessing.feature_builder import DataIngestion
from src.preprocessing.scaler import Scaler
from src.models.evaluate import ModelEvaluation


class TrainPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            logging.info("Training pipeline started")

            # 1. Data ingestion (your current feature builder)
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")

            # 2. Scaling and preprocessing
            scaler = Scaler()
            train_arr, test_arr, scaler_path = scaler.initiate_data_transformation(
                train_path,
                test_path
            )

            logging.info("Data transformation completed")

            # 3. Model evaluation and selection
            model_eval = ModelEvaluation()
            best_model_name, best_model_score = model_eval.initiate_model_evaluation(
                train_arr,
                test_arr
            )

            logging.info(
                f"Training completed | "
                f"Best Model: {best_model_name} | "
                f"ROC AUC: {best_model_score}"
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
