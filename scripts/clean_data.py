# scripts/clean_data.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# -----------------------------
# Load validated dataset
# -----------------------------
df = pd.read_csv("data/exoplanets_validated.csv")
print("Loaded:", df.shape)

# -----------------------------
# Keep only required columns
# -----------------------------
required_cols = [
    "pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper"
]
df = df[required_cols]

# -----------------------------
# Remove invalid values
# -----------------------------
df = df[(df["pl_rade"] > 0) & (df["pl_bmasse"] > 0)]

# -----------------------------
# Handle missing values
# -----------------------------
imputer = SimpleImputer(strategy="median")
df[required_cols] = imputer.fit_transform(df[required_cols])

# -----------------------------
# Feature engineering
# -----------------------------
df["pl_dens"] = df["pl_bmasse"] / (df["pl_rade"] ** 3)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv("data/exoplanets_cleaned.csv", index=False)
print("✔ Cleaned dataset saved")
