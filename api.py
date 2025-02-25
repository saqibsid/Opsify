from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

#model
with open("models/XGBoostRegressor.pkl",'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class Features(BaseModel):
    payment_sequential: int
    payment_installments: int
    payment_value: float
    price: float
    freight_value: float
    product_name_lenght: int
    product_description_lenght: int
    product_photos_qty: int
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float
    product_volume: float
    product_density: float
    price_per_volume: float


@app.get('/')
def home():
    return {"hellloo":"hi there!"}

@app.post("/predict/")
def predict(features: Features):
    df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(df)
    return {"prediction" : prediction.tolist()}

