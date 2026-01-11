from fastapi import FastAPI
from predict import predict_habitability

app = FastAPI(title="Exoplanet Habitability API")

@app.post("/predict")
def predict(data: dict):
    return predict_habitability(data)
