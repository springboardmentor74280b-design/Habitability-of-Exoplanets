# scripts/rank_exoplanets.py
import pandas as pd
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "exoplanets_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "final_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "artifacts", "ranked_exoplanets.csv")

# -----------------------------
# 1️⃣ Load cleaned data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2️⃣ Feature list (EXACTLY what model was trained on)
# -----------------------------
FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

# Safety check
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")

# -----------------------------
# 3️⃣ Load trained model
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# 4️⃣ Predict habitability score
# -----------------------------
df["habitability_score"] = model.predict_proba(df[FEATURES])[:, 1]

# -----------------------------
# 5️⃣ Rank exoplanets
# -----------------------------
df_ranked = df.sort_values("habitability_score", ascending=False)
df_ranked.insert(0, "rank", range(1, len(df_ranked) + 1))

# -----------------------------
# 6️⃣ Save ranking
# -----------------------------
df_ranked.to_csv(OUTPUT_PATH, index=False)

print("✔ Exoplanets ranked successfully")
print(f"✔ Saved to {OUTPUT_PATH}")

print("\nTop 5 ranked exoplanets:")
print(
    df_ranked[["rank", "pl_rade", "pl_eqt", "habitability_score"]]
    .head()
)
