from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predictDisease
from model.model import __version__ as model_version


app = FastAPI()

class TextIn(BaseModel):
    symptoms: str

class PredictionOut(BaseModel):
    final_prediction: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version":model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    disease = predictDisease(payload.symptoms)
    return {"disease":disease.final_prediction}