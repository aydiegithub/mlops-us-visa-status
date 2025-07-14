import os
import sys

import numpy as np
import pandas as pd

from usa_visa.entity.config_entity import USvisaPredictorConfig
from usa_visa.entity.r2_estimator import USvisaEstimator
from usa_visa.exception import AydieException
from usa_visa.logger import logging
from usa_visa.utils.main_utils import read_yaml_file
from pandas import DataFrame



class USvisaData:
    def __init__(self,
        continent,
        education_of_employee,
        has_job_experience,
        requires_job_training,
        no_of_employees,
        region_of_employment,
        prevailing_wage,
        unit_of_wage,
        full_time_position,
        company_age):
        """
        Constructor for the USvisaData class.

        This class is used to encapsulate all the input features required for making a US visa approval prediction.

        Args:
            continent (str): Continent where the job is located.
            education_of_employee (str): Education level of the employee.
            has_job_experience (bool): Whether the employee has prior job experience.
            requires_job_training (bool): Whether the job requires training.
            no_of_employees (int): Number of employees in the company.
            region_of_employment (str): Region within the country where the job is offered.
            prevailing_wage (float): Offered wage for the position.
            unit_of_wage (str): Wage unit (e.g., Hour, Year).
            full_time_position (bool): Whether the position is full-time.
            company_age (int): Age of the company
        """
        
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age
            logging.info("USvisaData object initialized with provided input features.")
        
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
    
    def get_usvisa_input_data_frame(self) -> DataFrame:
        """ 
        Dysfunction returns the data frame from USvisaData class input
        """
        logging.info("Entered [get_usvisa_input_data_frame] method of USvisaData class.")
        try:
            usvisa_input_data = self.get_usvisa_data_as_dict()
            logging.info("Converted input data to DataFrame.")
            return DataFrame(usvisa_input_data)
        
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
    
    def get_usvisa_data_as_dict(self):
        """ 
        This function returns a dictionary from USvisaData class input
        """
        logging.info("Entered [get_usvisa_data_as_dict] method of USvisaData class.")
        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }
            logging.info("Converted input data to dictionary.")
            return input_data
        except Exception as e:
            raise AydieException(e, sys) from e
        
    


class USvisaClassifier:
    def __init__(self, prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig()) -> None:
        """
        Constructor for the USvisaClassifier class.

        Initializes the USvisaClassifier with the provided prediction pipeline configuration.
        This configuration includes the model's bucket and file path used for predictions.

        Args:
            prediction_pipeline_config (USvisaPredictorConfig): Configuration object for model prediction.
        """
        logging.info("Entered [__init__] method of USvisaClassifier class.")
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            logging.info("USvisaClassifier object initialized with prediction pipeline config.")
        except Exception as e:
            raise AydieException(e, sys)
        
        
        
    def predict(self, dataframe) -> str:
        """
        Performs prediction using the US visa model stored in the specified S3 bucket.

        Args:
            dataframe (pd.DataFrame): Input dataframe containing features for prediction.

        Returns:
            str: Predicted visa approval status.
        """
        logging.info("Entered [predict] method of USvisaClassifier class.")
        try:
            model = USvisaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            logging.info("Model loaded successfully from R2 bucket.")
            result = model.predict(dataframe=dataframe)
            logging.info("Prediction completed.")
            return result
        except Exception as e:
            raise AydieException(e, sys)