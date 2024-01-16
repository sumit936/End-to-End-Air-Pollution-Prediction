import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models.values()))):

            #Fetching indivisual model
            model = list(models.values())[i]
            para = list(params.values())[i]

            #Applying GridsearchCv
            gs = GridSearchCV(model, para, cv = 3)
            gs.fit(x_train, y_train)

            #Trainig the model on best para

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            #Model prediction
            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)

            #Model score
            train_model_score  = r2_score(y_train,train_predict)
            test_model_score  = r2_score(y_test,test_predict)

            report[list(models.keys())[i]] = test_model_score
            return report
    
    except Exception as e:
        raise CustomException(e,sys)