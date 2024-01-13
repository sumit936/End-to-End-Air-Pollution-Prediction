import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass

class DataIngestionConfig:

    raw_data_path:str = os.path.join('artifacts','data.csv')
    train_path:str = os.path.join('artifacts','train.csv')
    test_path:str = os.path.join('artifacts','test.csv')

class DataIngestion:

    def __init__(self):     
        self.data_ingestion = DataIngestionConfig()
    
    def data_ingestion_initiated(self):

        try:

            df = pd.read_csv('notebook/data/airquality.csv')

            logging.info("Read the dataset!")
            os.makedirs(os.path.dirname(self.data_ingestion.train_path),exist_ok=True)

            df.to_csv(self.data_ingestion.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df,test_size=0.20, random_state=42)

            logging.info('Split the data into train and test!')

            train_set.to_csv(self.data_ingestion.train_path, index = False, header = True)            
            test_set.to_csv(self.data_ingestion.test_path, index = False, header = True)  

            return(

                self.data_ingestion.train_path,
                self.data_ingestion.test_path
            )          


        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":

    obj = DataIngestion()
    obj.data_ingestion_initiated()
