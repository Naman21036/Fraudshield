import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_source_path: str = os.path.join("data", "processed", "eda_done_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            df = pd.read_csv(self.ingestion_config.raw_source_path)
            logging.info("Dataset read successfully")

            if "Class" not in df.columns:
                raise CustomException("Target column 'Class' not found", sys)

            logging.info(f"Dataset shape: {df.shape}")
            logging.info(f"Class distribution:\n{df['Class'].value_counts()}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            train_set, test_set = train_test_split(
                df,
                test_size=0.3,
                random_state=42,
                stratify=df["Class"]
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
