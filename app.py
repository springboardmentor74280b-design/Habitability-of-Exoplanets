from flask import Flask, request, jsonify
from xgboost import XGBClassifier
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "..", "model", "scaler.pkl"))

CLASS_MAP = {
    0: "Non-Habitable",
    1: "Potentially Habitable",
    2: "Habitable"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])
    X_scaled = scaler.transform(df)

    pred = int(model.predict(X_scaled)[0])

    return jsonify({
        "class_id": pred,
        "class_name": CLASS_MAP[pred]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


