# backend/routes/ranking.py
import os
import pandas as pd
from flask import Blueprint, jsonify

ranking_bp = Blueprint("ranking", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RANK_FILE = os.path.join(BASE_DIR, "artifacts", "ranked_exoplanets.csv")

@ranking_bp.route("/api/rankings", methods=["GET"])
def get_rankings():
    if not os.path.exists(RANK_FILE):
        return jsonify({"error": "Ranking file not found"}), 404

    df = pd.read_csv(RANK_FILE)

    # Send only top 10 for UI
    top10 = df.head(10)

    return jsonify(top10.to_dict(orient="records"))
