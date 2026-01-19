import joblib
import pandas as pd
import os

BASE = os.path.dirname(__file__)

rf_model = joblib.load(os.path.join(BASE, "../models/xgboost_model.joblib"))

selected_features = pd.read_csv(
    os.path.join(BASE, "../data/selected_features.csv"),
    header=None
)[0].tolist()

top_exoplanets = pd.read_csv(
    os.path.join(BASE, "../data/exoplanets_preprocessed.csv")
).head(10)
