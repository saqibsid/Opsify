import json
import logging
# from .utils import get_data_for_test
import os

import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.ingest_data import ingest_df
from steps.clean_data import cleaning_data
from steps.model_train import model_training
from steps.evaluation import evaluate
from pydantic import BaseModel
from steps.config import ModelNameConfig
from steps.clean_data import PreProcessingData

docker_settings = DockerSettings(required_integrations=[MLFLOW]) # ensure that the pipeline runs inside a docker container

class DeploymentTriggerConfig(BaseModel): # change here to pydantic
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0

class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def dynamic_importer() -> str:
    """loads the data for test"""
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model"
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get active component from stack
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing serice with same pipeline name, step_name, and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name = model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    return accuracy > config.min_accuracy

#predictor pipeline
@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    service.start(timeout=10)
    data = json.loads(data)
    data.pop('columns')
    data.pop('index')
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_volume",
        "product_density",
        "price_per_volume",
    ]
    df = pd.DataFrame(data['data'],columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    # ingest data, clean, model train and evaluate the model
    df = ingest_df(datapath = "./data/processed_olist_customer_dataset.csv")
    X_train, X_test, y_train, y_test = cleaning_data(df)

    config = ModelNameConfig(model_name="LinearRegression")
    model = model_training(X_train, X_test, y_train, y_test,config)
    r2,rmse = evaluate(model,X_test,y_test)
    # decision to check if model to be deployed or not
    deployment_decision = deployment_trigger(accuracy=r2, config=DeploymentTriggerConfig(min_accuracy=min_accuracy))

    # if condition is met
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout
    )

# inference pipeline
@pipeline(enable_cache=False, settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        running=False
    )
    prediction = predictor(service=service, data=data)
    return prediction

def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df.sample(n=100)
        df = PreProcessingData(df)
        df.drop(["review_score"],axis=1,inplace=True)
        result = df.to_json(orient='split')
        return result
    except Exception as e:
        logging.error(e)
        raise e
