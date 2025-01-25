import logging
import pandas as pd
from zenml import step

@step
def cleaning_data(df:pd.DataFrame) -> None:
    """
    takes input as ingested data and cleans it
    """
    pass