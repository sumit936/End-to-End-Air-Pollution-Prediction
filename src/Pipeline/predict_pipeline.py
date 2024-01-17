import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object

class PredictPipeline():

    def __init__(self) -> None:
        pass

    def predict(self,features):
        logging.info("Entered prediction function!")
        try:
            model_path = os.path.join('artifacts','modle.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            # loading the model and preprocessor picle file
            logging.info("Loading the model and preprocessor pickle file..")
            model = load_object(os.path.dirname(model_path))
            preprocessor = load_object(os.path.dirname(preprocessor_path))
            
            logging.info('Pickle file loaded.')

            #transforming the data and predictig
            
            transformed_feature = preprocessor.transform(features)
            prediction = model.predict(transformed_feature)

            logging.info('Prediction Done!')
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    try:
        def __init__(self,
            Temperature:float, 
            Humidity:float, 
            Wind_Speed:float, 
            Visibility:float,
            Pressure:float, 
            so2:float, 
            no2:float, 
            Rainfall:float, 
            PM10:int, 
            PM25:float):

            self.Temperature = Temperature,
            self.Humidity = Humidity
            self.Wind_Speed = Wind_Speed
            self.Visibility = Visibility
            self.Pressure = Pressure
            self.so2 = so2
            self.no2 = no2
            self.Rainfall = Rainfall
            self.PM10 = PM10
            self.PM25 = PM25
        
        def get_data_as_data_frame(self):

            data = {'Temperature':[self.Temperature], 'Humidity':[self.Humidity], 'Wind.Speed..km.h.':[self.Wind_Speed], 'Visibility':[self.Visibility],
       'Pressure':[self.Pressure], 'so2':[self.so2], 'no2':[self.no2], 'Rainfall':[self.Rainfall], 'PM10':[self.PM10], 'PM25':[self.PM25]
       }
            return pd.DataFrame(data)
    except Exception as e:
        raise CustomException(e,sys)
        

        