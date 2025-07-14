from usa_visa.cloud_storage.cloudflareR2_storage import SimpleStorageService
from usa_visa.exception import AydieException
from usa_visa.entity.estimator import USvisaModel
import sys
from pandas import DataFrame
from usa_visa.logger import logging


class USvisaEstimator:
    """
    This class is used to save and retrieve usa_visas model in s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: USvisaModel = None
        logging.info(f"USvisaEstimator initialized with bucket: {bucket_name}, model_path: {model_path}")


    def is_model_present(self,model_path):
        try:
            logging.info(f"Checking if model exists at {model_path} in bucket {self.bucket_name}")
            result = self.s3.s3_key_path_available(bucket_name = self.bucket_name, s3_key = model_path)
            logging.info(f"Model presence check complete: {model_path}")
            return result
        except AydieException as e:
            print(e)
            return False

    def load_model(self,) -> USvisaModel:
        """
        Load the model from the model_path
        :return:
        """
        logging.info(f"Loading model from {self.model_path} in bucket {self.bucket_name}")
        return self.s3.load_model(self.model_path, bucket_name = self.bucket_name)

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            logging.info(f"Saving model from {from_file} to {self.model_path} in bucket {self.bucket_name}")
            self.s3.upload_file(from_file,
                                to_filename = self.model_path,
                                bucket_name = self.bucket_name,
                                remove = remove
                                )
            logging.info(f"Model successfully saved to {self.model_path}")
        except Exception as e:
            raise AydieException(e, sys)


    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            logging.info("Predict method called.")
            if self.loaded_model is None:
                logging.info("Model not loaded in memory. Loading model now...")
                self.loaded_model = self.load_model()
            result = self.loaded_model.predict(dataframe = dataframe)
            logging.info("Prediction completed successfully.")
            return result
        except Exception as e:
            raise AydieException(e, sys)