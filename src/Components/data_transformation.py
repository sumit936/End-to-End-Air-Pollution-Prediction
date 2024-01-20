import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from src.utils import save_object


@dataclass

class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

        def __init__(self):
            self.data_transformation_config = DataTransformationConfig()
        
        def outlier_handler(self,data):
            logging.info("Entered into outlier handler function")

            try:
                for feature_name in data.columns:
                    Q1 = data[feature_name].quantile(0.25)
                    Q3 =  data[feature_name].quantile(0.75)
                    
                    #Inter quantile range
                    IQR = Q3-Q1
                    
                    #setting upper and lower limits 
                    upper_limit = Q3+(1.5*IQR)
                    lower_limit = Q1-(1.5*IQR)
                    
                    data[feature_name] = np.where(data[feature_name]>upper_limit,upper_limit,
                                                np.where(data[feature_name]<lower_limit,lower_limit,data[feature_name]))
                logging.info('Outlier Removal Completed!')
                return data
            except Exception as e:
                raise CustomException(e, sys)
            
        def to_category(self, val):
            
            try:
                if val>0 and val<=50:
                    return 0
                    
                if val>50 and val<=100:
                    return 1
                
                if val>100 and val<=200:
                    return 2
                
                if val>200 and val<=300:
                    return 3
                
                if val>300 and val<=400:
                    return 4
                
                if val>400:
                    return 5
                
            except Exception as e:
                raise CustomException(e, sys)
            
        def get_data_transformation_obj(self):
        
            logging.info("Enterd into get data tranfromation obj function")

            """
            This is function is responsible for Data transformation
            """

            try:
                outlier_col = ['PM10', 'PM25', 'so2', 'no2']
                all_col = ['Temperature', 'Humidity', 'Wind.Speed..km.h.', 'Visibility',
       'Pressure', 'so2', 'no2', 'Rainfall', 'PM10', 'PM25']


                preprocessor = ColumnTransformer(
                    [
                        ('oh', FunctionTransformer(self.outlier_handler), outlier_col),
                        ('scaler', StandardScaler(), all_col)

                    ]
                )
                

                return preprocessor
        
            except Exception as e:
                raise CustomException(e, sys)
        
        def initiate_data_transformation(self, train_path, test_path):

            try:

                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Creating a New feature..")

                train_df['AQI (Category Range)'] = train_df['AQI'].apply(self.to_category)
                test_df['AQI (Category Range)'] = test_df['AQI'].apply(self.to_category)

                logging.info("New feature Created.")

                logging.info('Obtaining data preprocessing object..')


                preprocessing_obj = self.get_data_transformation_obj()

                logging.info('Object Obatained.')

                target_feature_name = 'AQI'
                classification_target_feature_name = 'AQI (Category Range)'
                
                input_feature_train = train_df.drop([target_feature_name,classification_target_feature_name],axis=1)
                target_feature_train = train_df[target_feature_name]

                input_feature_test = test_df.drop([target_feature_name,classification_target_feature_name], axis = 1)
                target_feature_test = test_df[target_feature_name]

                logging.info('Applying preprocessing object on training and testing dataframe')

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
                input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test)

                train_arr = np.c_[

                    input_feature_train_arr, np.array(target_feature_train)
                ]

                test_arr = np.c_[

                    input_feature_test_arr, np.array(target_feature_test)
                ]

                

                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessing_obj

                )
                logging.info('Preprocessing object Saved!')
                return(

                    train_arr, 
                    test_arr, 
                    self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomException(e, sys)
