import csv
import os
import json
import numpy as np
import pandas as pd
import joblib

from flask import (
    Flask, render_template, request,
    jsonify, redirect, url_for, send_file
)
from flask_cors import CORS

# ------------------------
# PATH CONFIGURATION
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ------------------------
# FLASK APP SETUP
# ------------------------
app = Flask(__name__)
CORS(app)

# ------------------------
# LOAD MODEL & SCALER
# ------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------
# FEATURE LIST
# ------------------------
FEATURES = [
    "P_RADIUS",
    "P_MASS",
    "P_TEMP_EQUIL",
    "P_PERIOD",
    "S_RADIUS",
    "S_TEMPERATURE"
]

# ------------------------
# SAVE TO CSV (SAFE)
# ------------------------
def save_to_csv(input_data, status):
    habitable = 100 if status == "Habitable" else 0
    not_habitable = 0 if status == "Habitable" else 100

    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(FEATURES + ["status", "habitable", "not_habitable"])
        writer.writerow(input_data + [status, habitable, not_habitable])

# ------------------------
# ROUTES
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

# ------------------------
# DASHBOARD
# ------------------------
@app.route("/dashboard")
def dashboard():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=FEATURES + ["status", "habitable", "not_habitable"])

    for col in FEATURES + ["habitable", "not_habitable"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    leaderboard = df.sort_values(by="habitable", ascending=False).head(10)

    return render_template(
        "dashboard.html",
        leaderboard=leaderboard.to_dict(orient="records"),
        total_predictions=len(df),
        total_habitable=len(df[df["status"] == "Habitable"]),
        total_not_habitable=len(df[df["status"] == "Not Habitable"]),
        avg_habitable=round(df["habitable"].mean(), 2) if not df.empty else 0,
        orbital_period=json.dumps(df["P_PERIOD"].tolist()),
        temperature=json.dumps(df["P_TEMP_EQUIL"].tolist()),
        planet_mass=json.dumps(df["P_MASS"].tolist()),
        planet_radius=json.dumps(df["P_RADIUS"].tolist()),
        habitable=json.dumps(df["habitable"].tolist())
    )

# ------------------------
# HISTORY (FIXED)
# ------------------------
@app.route("/history")
def history():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=FEATURES + ["status", "habitable", "not_habitable"])

    # SAFE numeric conversion (NO DROPS)
    for col in FEATURES + ["habitable", "not_habitable"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    df = df.reset_index().rename(columns={"index": "csv_index"})

    return render_template(
        "history.html",
        leaderboard=df.to_dict(orient="records")
    )

# ------------------------
# DELETE RECORD
# ------------------------
@app.route("/delete/<int:index>", methods=["POST"])
def delete_record(index):
    if not os.path.exists(CSV_PATH):
        return redirect(url_for("history"))

    df = pd.read_csv(CSV_PATH)

    if index in df.index:
        df = df.drop(index)

    df.to_csv(CSV_PATH, index=False)
    return redirect(url_for("history"))

# ------------------------
# RANKING
# ------------------------
@app.route("/ranking")
def ranking():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=FEATURES + ["status", "habitable", "not_habitable"])

    df["habitable"] = pd.to_numeric(df.get("habitable", 0), errors="coerce").fillna(0)
    df["status"] = df.get("status", "Not Habitable")

    df = df.sort_values(by="habitable", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return render_template("ranking.html", leaderboard=df.to_dict(orient="records"))

# ------------------------
# DOWNLOAD CSV
# ------------------------
@app.route("/download-csv")
def download_csv():
    if os.path.exists(CSV_PATH):
        return send_file(CSV_PATH, as_attachment=True)
    return "No CSV available", 404

# ------------------------
# API PREDICTION
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = [float(data[f]) for f in FEATURES]

    scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = int(model.predict(scaled)[0])

    status = "Habitable" if prediction == 1 else "Not Habitable"
    save_to_csv(input_data, status)

    return jsonify({
        "status": status,
        "habitable_percent": 100 if status == "Habitable" else 0,
        "not_habitable_percent": 0 if status == "Habitable" else 100
    })

# ------------------------
# FORM RESULT
# ------------------------
@app.route("/result", methods=["POST"])
def result():
    input_data = [float(request.form[f]) for f in FEATURES]

    scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = int(model.predict(scaled)[0])

    status = "Habitable" if prediction == 1 else "Not Habitable"
    save_to_csv(input_data, status)

    return render_template(
        "result.html",
        status=status,
        habitable=100 if status == "Habitable" else 0,
        not_habitable=0 if status == "Habitable" else 100
    )

# ------------------------
# RUN (LOCAL)
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
