import sys

from usa_visa.exception import AydieException
from usa_visa.logger import logging

from usa_visa.components.data_ingestion import DataIngestion
from usa_visa.components.data_validation import DataValidation
from usa_visa.components.data_transformation import DataTransformation

from usa_visa.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from usa_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of pipeline is responsible to start data ingestion

        Returns: DataIngestionArtifact
        object with trained and tested final file path
        """
        
        try:
            logging.info("Entered the [start_data_ingestion] method of TrainPipelin class")
            logging.info("Getting data from mongo db")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            
            return data_ingestion_artifact
            
        except Exception as e:
            raise AydieException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                                data_validation_config=self.data_validation_config
                                                )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )
            return data_validation_artifact

        except Exception as e:
            raise AydieException(e, sys) from e
        
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransformation(
                                        data_ingestion_artifact = data_ingestion_artifact,
                                        data_transformation_config = self.data_transformation_config,
                                        data_validation_artifact = data_validation_artifact
                                )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_ingestion_artifact
        except Exception as e:
            raise AydieException(e, sys)
        
        
        
    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running the complete training pipeline
        """
        try:
            logging.info("Entered the [run_pipeline] method of TrainPipelin class")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact = data_ingestion_artifact, data_validation_artifact = data_validation_artifact
            )
            logging.info("Exited the [run_pipeline] method of TrainPipelin class")
            
        except Exception as e:
            raise AydieException(e, sys) from e