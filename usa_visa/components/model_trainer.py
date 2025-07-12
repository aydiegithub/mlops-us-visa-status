import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from neuro_mf import ModelFactory

from usa_visa.exception import AydieException
from usa_visa.logger import logging
from usa_visa.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from usa_visa.entity.config_entity import ModelTrainerConfig
from usa_visa.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from usa_visa.entity.estimator import USvisaModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        """
        Args:
            data_transformation_artifact (DataTransformationArtifact): Output reference of data ingestion artifact
            model_trainer_config (ModelTrainerConfig): Configuration for data transformation
        """
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        
    
    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        This method uses neuro_mf to get the best model object and report of the best model

        Args:
            train (np.array): Input data for training
            test (np.array): Input data for testing

        Returns:
            Tuple[object, object]: Returns the metric artifact object and best model
        """
        logging.info("Entered [get_model_object_and_report] method of ModelTrainer class")
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path = self.model_trainer_config.model_config_file_path)
            
            X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            
            best_model_detail = model_factory.get_best_model(
                X = X_train, y = y_train, base_accuracy = self.model_trainer_config.expected_accuracy
            )
            
            model_obj = best_model_detail.best_model
            
            y_pred = model_obj.predict(X_test)
            
            accuracy_score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            logging.info("Exiting [get_model_object_and_report] method of ModelTrainer class")
            return best_model_detail, metric_artifact
            
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
        
    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        """
        This function initiates the model trainer steps
        Output: returns model trainer artifact
        """
        logging.info("Entered [initiate_model_trainer] method of ModelTrainer class")
        
        try:
            train_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_file_path)
            
            best_model_detail, metric_artifact = self.get_model_object_and_report(train = train_arr, test = test_arr)
            preprocessing_obj = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
            
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            
            usvisa_model = USvisaModel(preprocessing_object = preprocessing_obj, 
                                       trained_model_object = best_model_detail.best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path")
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                metric_artifact = metric_artifact
            )
            
            logging.info(f"model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise AydieException(e, sys) from e
        