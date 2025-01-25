import logging
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
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

    Returns:

    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
             raise ValueError(f"model {config.model_name} is not supported")
    except Exception as e:
        logging.error(f"Error while training the model {e}")
        raise e



def LinearRegressionModel(self,X_train,y_train,**kwargs):
    try:
        reg = LinearRegression(**kwargs)
        reg.fit(X_train,y_train)
        logging.info("Model training completed!")
        return reg
    except Exception as e:
        logging.error(f"Error while training the model {e}")
        raise e