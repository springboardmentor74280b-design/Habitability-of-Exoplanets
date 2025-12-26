# scripts/rank_exoplanets.py
import pandas as pd
import joblib

# 1️⃣ Load cleaned data
df = pd.read_csv("data/exoplanets_cleaned.csv")
features = ["pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper", "pl_dens", "st_teff", "st_rad", "st_mass", "st_lum"]

# 2️⃣ Load trained model
model = joblib.load("artifacts/final_model.pkl")

# 3️⃣ Predict habitability probabilities
df["habitability_score"] = model.predict_proba(df[features])[:,1]

# 4️⃣ Rank exoplanets
df_ranked = df.sort_values(by="habitability_score", ascending=False)
df_ranked.to_csv("artifacts/ranked_exoplanets.csv", index=False)

print("✔ Exoplanets ranked and saved in artifacts/ranked_exoplanets.csv")
