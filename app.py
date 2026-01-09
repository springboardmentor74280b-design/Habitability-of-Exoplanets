from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "..", "model", "scaler.pkl"))

# 🔹 MODEL FEATURES (MUST MATCH TRAINING)
FEATURES = [
    "P_MASS",
    "P_RADIUS",
    "P_PERIOD",
    "P_FLUX",
    "P_DISTANCE"
]

# 🔹 Frontend → Model mapping
FRONTEND_TO_MODEL = {
    "Mass (Earth Mass)": "P_MASS",
    "Radius (Earth Radii)": "P_RADIUS",
    "Orbital Period (Days)": "P_PERIOD",
    "Flux (Earth Flux)": "P_FLUX",
    "Distance to Star (AU)": "P_DISTANCE"
}

CLASS_MAP = {
    0: "Non-Habitable ❌",
    1: "Potentially Habitable 🌍"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # 🔹 Build row safely
    row = {}
    for frontend_key, model_key in FRONTEND_TO_MODEL.items():
        row[model_key] = data.get(frontend_key, 0)

    df = pd.DataFrame([row], columns=FEATURES)

    # 🔹 Scale + Predict
    X_scaled = scaler.transform(df)
    pred = int(model.predict(X_scaled)[0])

    return jsonify({
        "prediction": CLASS_MAP[pred]
    })

if __name__ == "__main__":
    app.run(debug=True)
