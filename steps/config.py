# from zenml.steps import BaseParameters
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    model_name: str = "XGBoostRegressor"
