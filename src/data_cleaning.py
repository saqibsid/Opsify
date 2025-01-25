import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd

import pandas as pd

class DataStrategy(ABC):
    """
    abstractive class for data cleaning. It acts as a blueprint for other classes
    """
    @abstractmethod
    def data_handling(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessing:

    def data_handling(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        preprocess the data
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
            logging.error(f"Error in preprocessing data {e}")
            raise e