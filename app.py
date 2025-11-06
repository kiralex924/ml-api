from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()  # ← СНАЧАЛА создаём app

@app.post("/debug_echo")
async def debug_echo(req: Request):
    raw = await req.body()
    return {"raw": raw.decode("utf-8", "ignore")}

# === дальше ваш основной код ===
model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

class PredictRequest(BaseModel):
    Pclass: int
    Age: float
    Fare: float
    SibSp: int = 0
    Parch: int = 0
    Sex_male: int = 0
    Embarked_S: int = 0
    Embarked_Q: int = 0

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    for c in columns:
        if c not in df.columns:
            df[c] = 0
    df = df[columns]
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}