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
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            # loading the model and preprocessor pickle file
            logging.info("Loading the model and preprocessor pickle file..")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            logging.info('Pickle file loaded.')

            #transforming the data and predictig
            
            transformed_feature = preprocessor.transform(features)
            prediction = model.predict(transformed_feature)

            logging.info('Prediction Done!')
            range = ''
            if prediction>0 and prediction<=50:
                range = 'Good'
                    
            if prediction>50 and prediction<=100:
                range = 'Satisfactory'

            if prediction>100 and prediction<=200:
                range = 'Moderate'

            if prediction>200 and prediction<=300:
                range = 'Poor'

            if prediction>300 and prediction<=400:
                range = 'Very Poor'

            if prediction>400:
                range = 'Severe'

            return (prediction,range)
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
            PM25:float
            ):

            self.Temperature = Temperature
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

            data = {'Temperature': [self.Temperature], 'Humidity':[self.Humidity], 'Wind.Speed..km.h.':[self.Wind_Speed], 'Visibility':[self.Visibility],
       'Pressure':[self.Pressure], 'so2':[self.so2], 'no2':[self.no2], 'Rainfall':[self.Rainfall], 'PM10':[self.PM10], 'PM25':[self.PM25]
       }
            return pd.DataFrame(data)
    except Exception as e:
        raise CustomException(e,sys)
    
# if __name__ == '__main__':
    
#     obj = CustomData(9.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9,10.0)
#     df = obj.get_data_as_data_frame()
#     print(df)
#     Preds = PredictPipeline()
#     pred = Preds.predict(df)
#     print(pred)
        

        