from usa_visa.entity.config_entity import ModelEvaluationConfig
from usa_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact

from sklearn.metrics import f1_score
from usa_visa.exception import AydieException
from usa_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from usa_visa.logger import logging

import sys
import pandas as pd

from typing import Optional

from usa_visa.entity.r2_estimator import USvisaEstimator
from usa_visa.entity.estimator import USvisaModel
from usa_visa.entity.estimator import TargetValueMapping

from dataclasses import dataclass


@dataclass
class EvaluationModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float
    


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise AydieException(e, sys) from e
        
    
    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        This function is used to get model in production

        Returns:
            Optional[USvisaEstimator]: Returns model is available in s3 bucket
        """
        
        try:
            logging.info("Fetching best model from production (S3).")
            bucket_name = self.model_eval_config.buck_name
            model_path = self.model_eval_config.s3_model_key_path
            usvisa_estimator = USvisaEstimator(bucket_name=bucket_name, model_path=model_path)
            
            if usvisa_estimator.is_model_present(model_path=model_path):
                logging.info(f"Model found at {model_path} in bucket {bucket_name}.")
                return usvisa_estimator
            
            logging.info(f"No model found at {model_path} in bucket {bucket_name}.")
            return None
        
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
    def evaluate_model(self) -> EvaluationModelResponse:
        """
         This function is used to evaluate trained model with the production model and choose the best model

        Output:
            returns bool value based on validation results
        """
        
        try:
            logging.info("Reading test data from parquet file.")
            test_df = pd.read_parquet(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            
            logging.info("Preparing features and target variable.")
            X, y = test_df.drop(TARGET_COLUMN, axis = 1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict)
            
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"Trained model F1 score: {trained_model_f1_score}")
            
            best_model_f1_score = None
            best_model = self.get_best_model()
            
            if best_model is not None:
                logging.info("Best model found in production. Calculating F1 score for best model.")
                y_hat_best_model = best_model.predict(X)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"Best model F1 score: {best_model_f1_score}")
            else:
                logging.info("No best model found in production.")
                
            temp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluationModelResponse(
                trained_model_f1_score = trained_model_f1_score,
                best_model_f1_score = best_model_f1_score,
                is_model_accepted = trained_model_f1_score > temp_best_model_score, 
                difference = trained_model_f1_score - temp_best_model_score
            )
            
            logging.info(f"Evaluation result: {result}")
            return result
        
        except Exception as e:
            raise AydieException(e, sys)
        
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        This function is used to initiate all steps of the model evaluation

        Returns:
            ModelEvaluationArtifact: Returns model evaluation artifact
        """
        try:
            logging.info("Starting model evaluation process.")
            evaluate_model_response = self.evaluate_model()
            logging.info(f"Model evaluation completed. Evaluation response: {evaluate_model_response}")
            s3_model_path = self.model_eval_config.s3_model_key_path
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted = evaluate_model_response.is_model_accepted,
                s3_model_path = s3_model_path,
                trained_model_path = self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy = evaluate_model_response.difference
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise AydieException(e, sys) from e