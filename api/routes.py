from flask import Blueprint, request, jsonify, render_template
import numpy as np
from .models import rf_model, selected_features, top_exoplanets

api_bp = Blueprint("api", __name__)

@api_bp.route("/")
def home():
    numeric = [f for f in selected_features if not f.startswith("st_spectype_")]
    return render_template("index.html", features=numeric)

@api_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    X = []
    for f in selected_features:
        if f.startswith("st_spectype_"):
            X.append(1 if f == data["spectral_type"] else 0)
        else:
            X.append(float(data.get(f, 0)))

    X = np.array([X])
    X = np.array([[data["P_PERIOD"], data["P_MASS"], data["P_TEMP_EQUIL"]]])


    prob = rf_model.predict_proba(X)[0][1]
    return jsonify({
        "habitability": int(prob > 0.5),
        "probability": round(float(prob), 3)
    })
