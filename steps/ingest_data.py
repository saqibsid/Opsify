## python file containing steps to ingest the data
import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from data_path
    """
    def __init__(self,data_path : str):
        """
        initialising the data_path

        Args:
            data_path : path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from data_path
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
# calling the class in a pipeline step
@step
def ingest_df(datapath:str) -> pd.DataFrame:
    """
    ingesting the data from data_path

    Args:
        data_path : path to the data
    Returns:
        pd.Dataframe: the ingested data
    """
    try:
        ingest_data = IngestData(datapath)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data {e}")
        raise e