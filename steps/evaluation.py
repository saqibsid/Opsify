import logging
import pandas as pd
from zenml import step
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate(model:RegressorMixin,
             X_test : pd.DataFrame,
             y_test : pd.DataFrame)-> Tuple[
                 Annotated[float,"r2"],
                 Annotated[float,"rmse"]
             ]:
    try:
        predictions = model.predict(X_test)
        mse = MSE(y_test,predictions)
        mlflow.log_metric("mse",mse)
        r2 = R2(y_test,predictions)
        mlflow.log_metric("r2",r2)
        rmse = RMSE(y_test,predictions)
        mlflow.log_metric("rmse",rmse)

        return r2,rmse
    except Exception as e:
        logging.error(f"Error while evaluating scores {e}")
        raise e

def MSE(y_true:np.ndarray,y_pred:np.ndarray) -> float:
    try:
        logging.info("Calculating MSE")
        mse = mean_squared_error(y_true=y_true,y_pred=y_pred)
        logging.info(f"MSE: {mse}")
        return mse
    except Exception as e:
        logging.error(f"Error while calculating MSE {e}")
        raise e
    
def R2(y_true:np.ndarray,y_pred:np.ndarray) -> float:
    try:
        logging.info("Calculating R2 score")
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        logging.info(f"R2: {r2}")
        return r2
    except Exception as e:
        logging.error(f"Error while calculating R2 score {e}")
        raise e
    
def RMSE(y_true:np.ndarray,y_pred:np.ndarray) -> float:
    try:
        logging.info("Calculating RMSE score")
        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        logging.info(f"R2: {rmse}")
        return rmse
    except Exception as e:
        logging.error(f"Error while calculating rmse score {e}")
        raise e