from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

class Passenger(BaseModel):
    Sex: str
    Age: float
    Fare: float

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Passenger):
    df = pd.DataFrame([{
        'Sex_male': 1 if data.Sex == 'male' else 0,
        'Age': data.Age,
        'Fare': data.Fare
    }])
    for c in columns:
        if c not in df.columns:
            df[c] = 0
    df = df[columns]
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
