import logging
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client # allows access to backend to manage artifacts, track experiments etc..


experiment_tracker = Client().active_stack.experiment_tracker # connect with backend and get the active artifacts

@step(experiment_tracker = experiment_tracker.name) # so that this step will be tracked
def model_training(X_train : pd.DataFrame,
                   X_test : pd.DataFrame,
                   y_train : pd.DataFrame,
                   y_test : pd.DataFrame,
                   config: ModelNameConfig) -> RegressorMixin:
    """
    training the model

    Args:
        X_train : Training Data,
        X_test : Testing Data,
        y_train : Training Labels,
        y_test : Testing Labels
        config: Configuration for model selection

    Returns:
        trained_model : model trained on data
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel(X_train,y_train)
            return model
        else:
             raise ValueError(f"model {config.model_name} is not supported")
    except Exception as e:
        logging.error(f"Error while training the model {e}")
        raise e

# done upto here do evaluation part -> 1:40:15

def LinearRegressionModel(X_train,y_train,**kwargs):
    try:
        reg = LinearRegression(**kwargs)
        reg.fit(X_train,y_train)
        logging.info("Model training completed!")
        return reg
    except Exception as e:
        logging.error(f"Error while training the model {e}")
        raise e