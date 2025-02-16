from fastapi import FastAPI
import pickle
import pandas as pd

#model
with open("models/XGBoostRegressor.pkl",'rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get('/')
def home():
    return {"hellloo":"hi there!"}

@app.post("/predict/")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return {"prediction" : prediction.tolist()}

