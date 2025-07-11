import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from usa_visa.entity.config_entity import DataIngestionConfig
from usa_visa.entity.artifact_entity import DataIngestionArtifact
from usa_visa.exception import AydieException
from usa_visa.logger import logging
from usa_visa.data_access.usvisa_data import USvisaData

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        logging.info("Data_Ingestion class object created")
        
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        
        logging.info("Entered [export_data_into_feature_store] method of Data_Ingestion class")
        
        try:
            logging.info(f"Exporting data from mongodb")
            usvisa_data = USvisaData()
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name = self.data_ingestion_config.collection_name
                )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            
            if dataframe.empty:
                logging.error("Fetched dataframe from MongoDB is empty. Cannot proceed.")
                raise Exception("No data found in the MongoDB collection.")
            
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok = True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_parquet(feature_store_file_path, index = False)
            return dataframe
        
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Train and test Parquet files saved locally
        On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Entered [split_data_as_train_test] method of DataIngestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            logging.info("Performed train test split operation on the dataframe")

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.test_file_path

            train_dir = os.path.dirname(train_path)
            test_dir = os.path.dirname(test_path)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            logging.info(f"Exporting train file to {train_path}")
            train_set.to_parquet(train_path, index=False)

            logging.info(f"Exporting test file to {test_path}")
            test_set.to_parquet(test_path, index=False)

            logging.info("Exported train and test files successfully")

        except Exception as e:
            raise AydieException(e, sys) from e
            
            
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        
        logging.info("Entered [initiate_data_ingestion] method of DataIngestion class")
        
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from mongodb")
            
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")
            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path = self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.test_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise AydieException(e, sys) from e