from zenml.client import Client
from zenml import pipeline
import logging
import pandas as pd
from zenml import step
import numpy as np
from typing import Union, Annotated, Tuple
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# ingesting data
def ingest_df(datapath:str) -> pd.DataFrame:
    """
    ingesting the data from data_path

    Args:
        data_path : path to the data
    Returns:
        pd.Dataframe: the ingested data
    """
    try:
        df = pd.read_csv(datapath)
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data {e}")
        raise e

def cleaning_data(data:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.Series,"X_test"],
    Annotated[pd.Series,"y_test"]
]:
    """
    takes input as ingested data and cleans it and return the divided data

    Args:
        df : raw data
    Returns:
        X_train : Training Data
        X_test : Testing Data
        y_train : training labels
        y_test : testing labels
    """
    try:
        data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
        data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
        data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
        data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
        data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
        # write "No review" in review_comment_message column
        data["review_comment_message"].fillna("No review", inplace=True)

        data['product_volume'] = data['product_length_cm'] * data['product_height_cm'] * data['product_width_cm']
        data['product_density'] = data['product_weight_g'] / data['product_volume']
        data['price_per_volume'] = data['price']/data['product_volume']

        data = data.select_dtypes(include=[np.number])
        cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
        data = data.drop(cols_to_drop, axis=1)

        X = data.drop("review_score",axis=1)
        y = data["review_score"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        logging.info("Data cleaned completed!")
        print("asdasd",X_train.shape)
        print("asdasd",X_test.shape)
        print("asdasd",y_train.shape)
        print("asdasd",y_test.shape)
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f"Error in cleaning data {e}")
        raise e

def model_training(X_train : pd.DataFrame,
                   y_train : pd.DataFrame,
                   model_name: str) -> RegressorMixin:
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
        if model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            reg = LinearRegression()
            reg.fit(X_train,y_train)
            logging.info("Model training completed!")
            return reg
        else:
             raise ValueError(f"model {model_name} is not supported") 
    except Exception as e:
        logging.error(f"Error while training the model {e}")
        raise e

def evaluate(model:RegressorMixin,
             X_test : pd.DataFrame,
             y_test : pd.DataFrame)-> Tuple[
                 Annotated[float,"r2"],
                 Annotated[float,"rmse"]
             ]:
    try:
        predictions = model.predict(X_test)
        r2 = r2_score(y_test,predictions)
        mlflow.log_metric("r2",r2)
        logging.info(f"R2: {r2}")
        rmse = root_mean_squared_error(y_test,predictions)
        mlflow.log_metric("rmse",rmse)

        return r2,rmse
    except Exception as e:
        logging.error(f"Error while evaluating scores {e}")
        raise e

def training_pipelines(data_path:str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = cleaning_data(df)

    trained_model = model_training(X_train, y_train,model_name="LinearRegression")
    r2,rmse = evaluate(trained_model,X_test,y_test)

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipelines(data_path='data/olist_customers_dataset.csv')