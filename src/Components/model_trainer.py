import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_model
from sklearn.metrics import r2_score

from sklearn.ensemble import (

    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:

    model_obj_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():

    def __init__(self):
       
       self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_ar, test_ar):

        try:

            logging.info("Splitting the training and testing array...")

            x_train, y_train, x_test, y_test = (

                train_ar[:,:-1],
                train_ar[:,-1],
                test_ar[:,:-1],
                test_ar[:,-1]
            )

            models = {

                "LinearRegressor":LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "GradientBoostRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "KNNRegressor": KNeighborsRegressor(),
                
            }

            params = {

                "Linear Regression":{},

                "Random Forest":{
                        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
                        # 'max_features':['sqrt','log2',None],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Decision Tree": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    },
                    
                    "Gradient Boosting":{
                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        # 'criterion':['squared_error', 'friedman_mse'],
                        # 'max_features':['auto','sqrt','log2'],
                        'n_estimators': [8,16,32,64,128,256]
                    },

                    "AdaBoost Regressor":{
                        'learning_rate':[.1,.01,0.5,.001],
                        # 'loss':['linear','square','exponential'],
                        'n_estimators': [8,16,32,64,128,256]
                    },

                    "KNearsetNeighbors Regressor":{

                        'n_neighbors': [8,16,32,64,128,256]
                    }

            }

            #Model Report
            model_report:dict = evaluate_model(x_test=x_test, x_train=x_train, y_train=y_train, y_test=y_test, models=models,params=params)

            #get the best model score
            best_model_score = max(sorted(list(model_report.values())))

            #get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            if best_model_score<0.7:
                raise CustomException("No best Model found!")
            
            best_model = models[best_model_name]
            
            save_object(

                self.model_trainer_config.model_obj_file_path,
                best_model
            )

            predicton = best_model.predict(x_test)
            r2score = r2_score(y_test, predicton)
            

            return r2score
        
        except Exception as e:
            raise CustomException(e, sys)
    
