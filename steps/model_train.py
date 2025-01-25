import logging
import pandas as pd
from zenml import step


@step
def model_training(df : pd.DataFrame) -> None:
    pass

