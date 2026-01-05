# backend/routes/predict.py
import os
import joblib
import pandas as pd
from flask import Blueprint, request, jsonify

predict_bp = Blueprint("predict", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "final_model.pkl")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        pl_rade = float(data["radius"])
        pl_bmasse = float(data["mass"])
        pl_eqt = float(data["temperature"])
        pl_orbper = float(data["orbital_period"])

        pl_dens = pl_bmasse / (pl_rade ** 3)

        input_df = pd.DataFrame([[
            pl_rade,
            pl_bmasse,
            pl_eqt,
            pl_orbper,
            pl_dens
        ]], columns=FEATURES)

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        return jsonify({
            "habitability_prediction": prediction,
            "habitability_label": "Habitable" if prediction == 1 else "Non-Habitable",
            "habitability_score": round(probability, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
