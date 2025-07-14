import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"

MONGODB_URL_KEY = "mongodb_conn"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.parquet"
TEST_FILE_NAME: str = "test.parquet"

FILE_NAME: str = "usvisa.parquet"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH_URL = os.path.join("config", "schema.yaml")

#############################################################################################
# AWS CREDENTIAL
'''AWS_ACCESS_KEY_ID_ENV_KEY  = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"'''

#############################################################################################

# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

#############################################################################################

# Data Validation Constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

#############################################################################################

# Data Transoformation Constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

#############################################################################################

# Model Trainer Constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

#############################################################################################

'''# Model Evaluation Constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME: str = "usvisa-model2025"
MODEL_PUSHER_S3_KEY: str = "model-registory"'''

#############################################################################################

# Cloudflare R2 CREDENTIALS
R2_ACCOUNT_ID_ENV_KEY = "R2_ACCOUNT_ID"
R2_ACCESS_KEY_ID_ENV_KEY = "R2_ACCESS_KEY_ID"
R2_SECRET_ACCESS_KEY_ENV_KEY = "R2_SECRET_ACCESS_KEY"
R2_BUCKET_NAME = "usvisa-model"
R2_ENDPOINT_URL = "R2_ENDPOINT"

# Model Evaluation Constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME: str = "usvisa-model"
MODEL_PUSHER_S3_KEY: str = "model-registory"