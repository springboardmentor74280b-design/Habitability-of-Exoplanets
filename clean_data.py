import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1️⃣ Load validated dataset
# -----------------------------
df = pd.read_csv("data/exoplanets_validated.csv")
print("\nLoaded Validated Dataset:", df.shape)

# -----------------------------
# 2️⃣ Handle Missing Values
# -----------------------------
# Automatically select all numeric columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Exclude pl_dens and st_lum (we calculate them manually)
num_cols = [c for c in num_cols if c not in ['pl_dens', 'st_lum']]

# Impute numeric columns with median
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Automatically select all categorical columns (object type)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute categorical columns with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("✔ Missing values handled for all columns")

# -----------------------------
# 3️⃣ Remove impossible values
# -----------------------------
df = df[df["pl_rade"] > 0]
df = df[df["pl_bmasse"] > 0]

# -----------------------------
# 4️⃣ Feature Engineering
# -----------------------------
# Planet density: mass / radius^3
df["pl_dens"] = df.apply(
    lambda row: row["pl_bmasse"] / (row["pl_rade"] ** 3) if pd.isnull(row["pl_dens"]) else row["pl_dens"],
    axis=1
)

# Stellar luminosity: radius^2 * teff^4
df["st_lum"] = df.apply(
    lambda row: (row["st_rad"] ** 2) * (row["st_teff"] ** 4) if pd.isnull(row["st_lum"]) else row["st_lum"],
    axis=1
)

# -----------------------------
# 5️⃣ Normalize numerical features
# -----------------------------
scaled_cols = ["pl_rade", "pl_bmasse", "pl_eqt", "pl_insol", "st_teff", "st_rad", "pl_dens", "st_lum"]

# Ensure only columns that exist in the dataframe are scaled
scaled_cols = [col for col in scaled_cols if col in df.columns]

scaler = MinMaxScaler()
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

print("✔ Normalization complete")

# -----------------------------
# 6️⃣ Save cleaned dataset
# -----------------------------
df.to_csv("data/exoplanets_cleaned.csv", index=False)
print("\n✔ Cleaned dataset saved as exoplanets_cleaned.csv")
