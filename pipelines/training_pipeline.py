from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import cleaning_data
from steps.model_train import model_training
from steps.evaluation import evaluate
from steps.config import ModelNameConfig

@pipeline
def training_pipelines(data_path:str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = cleaning_data(df)
    
    config = ModelNameConfig(model_name="LinearRegression")
    trained_model = model_training(X_train, X_test, y_train, y_test,config)
    r2,rmse = evaluate(trained_model,X_test,y_test)
