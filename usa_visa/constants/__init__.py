import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"

MONGODB_URL_KEY = os.getenv("mongodb_conn")

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.parquet"
TEST_FILE_NAME: str = "test.parquet"

FILE_NAME: str = "usvisa.parquet"

MODEL_FILE_NAME = "model.pkl"


# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME: str = "us_visa"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2