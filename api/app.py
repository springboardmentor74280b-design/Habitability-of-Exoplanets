from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)


# LOAD MODEL & FILES 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "selected_features.csv")
TOP_PLANETS_PATH = os.path.join(BASE_DIR, "data", "top_exoplanets.csv")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

#features = pd.read_csv(FEATURES_PATH)["feature"].tolist()
features = pd.read_csv(FEATURES_PATH).iloc[:, 0].tolist()

top_planets = pd.read_csv(TOP_PLANETS_PATH).head(10).to_dict(orient="records")

#routes

@app.route("/")
def index():
    return render_template(
        "index.html",
        features=features,
        top_planets=top_planets
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        #X = np.array([data[f] for f in features]).reshape(1, -1)
        X = np.array([float(data.get(f, 0)) for f in features]).reshape(1, -1)

        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob >= 0.5)

        return jsonify({
            "success": True,
            "habitability": pred,
            "probability": round(float(prob), 4)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)



