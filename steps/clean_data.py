import logging
import pandas as pd
from zenml import step
import numpy as np
from typing import Union, Annotated, Tuple
from sklearn.model_selection import train_test_split

@step
def cleaning_data(df:pd.DataFrame) -> Tuple[
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
        cleaned_data = PreProcessingData(data=df)
        X_train, X_test, y_train, y_test = DivideData(cleaned_data)
        logging.info("Data cleaned completed!")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f"Error in cleaning data {e}")
        raise e


def PreProcessingData(data:pd.DataFrame) -> pd.DataFrame:
    """
    function to preprocess the data
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

        data = data.select_dtypes(include=[np.number])
        cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
        data = data.drop(cols_to_drop, axis=1)

        return data
    except Exception as e:
        logging.error(f"Error while Preprocessing the data {e}")
        raise e
    
def DivideData(data:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]]:
    """
    defining the independent and target variable.
    """
    try:
        X = data.drop("review_score",axis=1)
        y = data["review_score"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error while dividing the data {e}")
        raise e