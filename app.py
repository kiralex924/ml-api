from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()  # ← сначала создаём приложение

# ---------------------------------------
# Эндпоинт для проверки сырых данных
# ---------------------------------------
@app.post("/debug_echo")
async def debug_echo(req: Request):
    raw = await req.body()
    return {"raw": raw.decode("utf-8", "ignore")}
# ---------------------------------------

# Загружаем модель
model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

# Модель данных
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
        "Sex_male": 1 if data.Sex == "male" else 0,
        "Age": data.Age,
        "Fare": data.Fare
    }])
    for c in columns:
        if c not in df.columns:
            df[c] = 0
    df = df[columns]
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
