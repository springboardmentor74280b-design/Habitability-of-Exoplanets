# backend/feature_ranges.py
import pandas as pd

# -----------------------------
# Load datasets
# -----------------------------
df_clean = pd.read_csv("../data/exoplanets_cleaned.csv")
df_raw = pd.read_csv("../data/exoplanets_validated.csv")

print("\n--- CLEANED DATA FEATURE STATISTICS ---")
print(df_clean.describe())

# -----------------------------
# Create habitability label from RAW data
# -----------------------------
habitable = (
    (df_raw["pl_rade"] < 2.5) &
    (df_raw["pl_eqt"] > 180) &
    (df_raw["pl_eqt"] < 400)
).astype(int)

# -----------------------------
# Attach label ONLY for analysis
# -----------------------------
df_clean["habitable"] = habitable.values

print("\n--- HABITABILITY CLASS DISTRIBUTION (RAW-BASED) ---")
print(df_clean["habitable"].value_counts())

# -----------------------------
# Feature-wise comparison
# -----------------------------
features = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_lum"
]

print("\n--- FEATURE MEANS BY CLASS ---")
print(df_clean.groupby("habitable")[features].mean())

print("\n--- FEATURE MEDIANS (USED BY BACKEND DEFAULTS) ---")
print(df_clean[features].median())
