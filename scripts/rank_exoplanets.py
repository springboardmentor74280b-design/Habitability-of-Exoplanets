# scripts/rank_exoplanets.py
import pandas as pd
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RAW_PATH = os.path.join(DATA_DIR, "exoplanets_validated.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "exoplanets_cleaned.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "ranked_exoplanets.csv")

# -----------------------------
# Load datasets
# -----------------------------
df_raw = pd.read_csv(RAW_PATH)
df_clean = pd.read_csv(CLEAN_PATH)

# -----------------------------
# Ensure planet name exists
# -----------------------------
if "pl_name" not in df_raw.columns:
    raise ValueError("❌ pl_name column missing in raw dataset")

# -----------------------------
# Reset index to align rows safely
# -----------------------------
df_raw = df_raw.reset_index(drop=True)
df_clean = df_clean.reset_index(drop=True)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# EXACT features used in training
# -----------------------------
FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

# -----------------------------
# Predict habitability score
# -----------------------------
df_clean["habitability_score"] = model.predict_proba(
    df_clean[FEATURES]
)[:, 1]

# -----------------------------
# Attach planet names SAFELY (INDEX-ALIGNED)
# -----------------------------
df_clean["pl_name"] = df_raw.loc[df_clean.index, "pl_name"].values

# -----------------------------
# Sort by habitability
# -----------------------------
df_ranked = df_clean.sort_values(
    by="habitability_score",
    ascending=False
)

# -----------------------------
# Final UI-ready output
# FRONTEND EXPECTS: pl_name
# -----------------------------
df_ranked_ui = df_ranked[
    ["pl_name", "pl_rade", "pl_eqt", "habitability_score"]
]

# -----------------------------
# Save
# -----------------------------
df_ranked_ui.to_csv(OUTPUT_PATH, index=False)

print("✅ Ranked exoplanets saved successfully")
print("📁", OUTPUT_PATH)
