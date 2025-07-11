import sys

from usa_visa.exception import AydieException
from usa_visa.logger import logging
from usa_visa.components.data_ingestion import DataIngestion
from usa_visa.entity.config_entity import DataIngestionConfig
from usa_visa.entity.artifact_entity import DataIngestionArtifact

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
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
        