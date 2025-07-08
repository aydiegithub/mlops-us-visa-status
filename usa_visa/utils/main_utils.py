import os
import sys

import numpy as np
import pandas as pd
import dill
import yaml

from usa_visa.exception import AydieException
from usa_visa.logger import logging



# This is custom function to read the yaml files
def read_yaml_file(file_path: str) -> dict:
    logging.info("Entered the read_yaml_file method of utils")
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
        
        logging.info("Exited the write_yaml_file method of utils")

    except Exception as e:
        raise AydieException(e, sys) from e
    



# This is custom function to write the yaml files
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    logging.info("Entered the write_yaml_file method of utils")
    try:
        if replace: 
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)
        
        logging.info("Exited the write_yaml_file method of utils")
            
    except Exception as e:
        raise AydieException(e, sys) from e
  
  
    

# This is used to load the custom objects
def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
            
        logging.info("Exited the load_object method of utils")
        return obj
            
    except Exception as e:
        raise AydieException(e, sys) from e
    
    
    
 
# This is used to save the custom objects
def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info("Exited the save_object method of utils")
    
    except Exception as e:
        raise AydieException(e, sys) from e


    

# This is used to drop specified columns from a DataFrame
def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    logging.info("Entered the drop_columns method of utils")
    
    try:
        df_dropped = df.drop(columns=cols, axis=1)
        logging.info(f"Dropped columns: {cols}")
        return df_dropped
    
    except Exception as e:
        raise AydieException(e, sys) from e




# This is used to load numpy array data from a file
def load_numpy_array_data(file_path: str) -> np.array:
    logging.info("Entered the load_numpy_array_data method of utils")
    
    try:
        with open(file_path, 'rb') as file_obj:
            array = np.load(file_obj)
        logging.info("Exited the load_numpy_array_data method of utils")
        return array
    
    except Exception as e:
        raise AydieException(e, sys) from e