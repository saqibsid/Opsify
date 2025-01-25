from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import cleaning_data
from steps.model_train import model_training
from steps.evaluation import evaluate


@pipeline
def training_pipelines(data_path:str):
    df = ingest_df(data_path)
    cleaning_data(df)
    model_training(df)
    evaluate(df)
