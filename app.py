# app.py — AstroHab Backend (Render + Local Compatible)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import traceback

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)

# ================= MODEL LOADING =================
print("\n" + "=" * 60)
print("🚀 LOADING ASTROHAB XGBOOST MODEL")
print("=" * 60)

model = None
preprocessor = None
feature_columns = []

try:
    model_files = {
        "model": "saved_models/xgb_habitability_model_0.pkl",
        "preprocessor": "saved_models/preprocessor.pkl",
        "features": "saved_models/feature_columns.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"✅ Found {name}: {path}")
        else:
            print(f"❌ Missing {name}: {path}")

    if os.path.exists(model_files["model"]):
        model = joblib.load(model_files["model"])
        print("✅ XGBoost model loaded")

    if os.path.exists(model_files["preprocessor"]):
        preprocessor = joblib.load(model_files["preprocessor"])
        print("✅ Preprocessor loaded")

    if os.path.exists(model_files["features"]):
        feature_columns = joblib.load(model_files["features"])
        print(f"✅ Feature columns loaded ({len(feature_columns)})")

except Exception as e:
    print("❌ Model loading failed")
    traceback.print_exc()

print("=" * 60)

# ================= CONSTANTS =================
CLASS_LABELS = {
    0: "Non-Habitable",
    1: "Potentially Habitable",
    2: "Highly Habitable",
}

# ================= HELPERS =================
def prepare_input(data):
    input_dict = {col: 0.0 for col in feature_columns}

    for key, value in data.items():
        key = key.lower()
        for col in feature_columns:
            if key in col.lower():
                try:
                    input_dict[col] = float(value)
                except:
                    pass

    return pd.DataFrame([input_dict])[feature_columns]

# ================= ROUTES =================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "features": len(feature_columns),
        "port": os.environ.get("PORT", "dynamic"),
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.json
        if not data:
            return jsonify({"error": "No input provided"}), 400

        df = prepare_input(data)

        X = preprocessor.transform(df) if preprocessor else df.values
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        return jsonify({
            "success": True,
            "prediction": prediction,
            "label": CLASS_LABELS[prediction],
            "confidence": round(float(probabilities[prediction] * 100), 2),
            "probabilities": {
                "Non_Habitable": round(float(probabilities[0] * 100), 2),
                "Potentially_Habitable": round(float(probabilities[1] * 100), 2),
                "Highly_Habitable": round(float(probabilities[2] * 100), 2),
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================= ENTRY POINT =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting AstroHab backend on port {port}")
    app.run(host="0.0.0.0", port=port)
