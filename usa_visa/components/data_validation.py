import json
import sys

import pandas as pd
from pandas import DataFrame

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from usa_visa.exception import AydieException
from usa_visa.logger import logging
from usa_visa.utils.main_utils import read_yaml_file, write_yaml_file
from usa_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from usa_visa.entity.config_entity import DataValidationConfig
from usa_visa.constants import SCHEMA_FILE_PATH_URL



class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Args:
            data_ingestion_artifact (DataIngestionArtifact): output reference of data injection artifact stage
            data_validation_config (DataValidationConfig): configuration for data validation
        """
        
        try:
            logging.info("Entered DataValidation constructor")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH_URL)
        except Exception as e:
            raise AydieException(e, sys)
        

        
    def validation_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Summary:
            This method validates the number of columns

        Args:
            dataframe (DataFrame): Input is pandas dataframe

        Returns:
            bool: returns bool value based on results
        """
        
        try:
            status = len(dataframe) == len(self._schema_config['columns'])
            logging.info(f"Required column is present: [{status}]")
            return status
        except Exception as e:
            raise AydieException(e, sys)
        
        
    
    
    def is_column_exist(self, dataframe: DataFrame) -> bool:
        """    
        Summary:
            This method validates column exists

        Args:
            dataframe (DataFrame): Input is pandas dataframe

        Returns:
            bool: returns bool value based on results
        """
        
        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            logging.info("checking for missing numerical columns")
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
                    
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
                
            
            logging.info("checking for missing categorical columns")
            for column in self._schema_config['categorical_columns']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
                    
            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: [{missing_categorical_columns}]")
                
            return False if len(missing_numerical_columns) > 0 or len(missing_numerical_columns) > 0 else True
                
        except Exception as e:
            raise AydieException(e, sys) from e

        

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise AydieException(e, sys)
        
    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame, ) -> bool:
        """
        This method validates if drift is detected

        Returns:
            bool: Return bool value based on validation result
        """
        
        try:
            logging.info('checking for drift in [detect_dataset_drift] method of DataValidation class')
            data_drift_profile = Profile(sections = [DataDriftProfileSection()])
            data_drift_profile.calculate(reference_data = reference_df, current_data = current_df)
            
            report = data_drift_profile.json()
            json_report = json.loads(report)
            
            write_yaml_file(file_path = self.data_validation_config.drift_report_file_path, content = json_report)
            
            n_features = json_report['data_drift']['data']['metrics']['_n_features']
            n_drifted_features = json_report['data_drift']['data']['metrics']['n_drifted_features']
            
            logging.info(f"{n_drifted_features} / {n_features} drift detected")
            drift_status = json_report['data_drift']['data']['metrics']['dataset_drift']
            return drift_status
            
        except Exception as e:
             raise AydieException(e, sys) from e
         
         
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        This method initiates data validation component for the pipeline
        """
        
        try:
            validation_error_msg = ""
            logging.info("Entered [initiate_data_validation] method of DataValidation class")
            logging.info("Starting data validation")
            
            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            status = self.validation_number_of_columns(dataframe = train_df)
            logging.info(f"All required number of columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in the training dataframe. "        
            
            status = self.validation_number_of_columns(dataframe = test_df)
            logging.info(f"All required  number of columns present in test dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in the test dataframe. "
                
            status = self.is_column_exist(train_df)
            logging.info(f"All required columns exist in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns[names] are missing in the train dataframe. "
                
            status = self.is_column_exist(test_df)
            logging.info(f"All required columns exist in test dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns[names] are missing in the test dataframe. "
                
            
            validation_status = len(validation_error_msg) == 0
            
            if validation_status:
                logging.info(f"Check for data drift")
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Data Drift detected")
                    validation_error_msg = "Drift_detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"validation_error: {validation_error_msg}")
                
                
            data_validation_artifact = DataValidationArtifact(
                validation_status = validation_status,
                message = validation_error_msg,
                drift_report_file_path = self.data_validation_config.drift_report_file_path 
            )
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
                    
        except Exception as e:
            raise AydieException(e, sys) from e