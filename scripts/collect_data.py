import pandas as pd

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv("data/exoplanets_raw.csv")  # relative path from project root

# -----------------------------
# 2️⃣ Show basic info
# -----------------------------
print("\n--- Dataset Loaded ---")
print(df.head())

# Print number of rows and columns
print("\nDataset Shape:", df.shape)

# -----------------------------
# 3️⃣ Check required columns exist
# -----------------------------
required_columns = [
    "pl_name", "hostname",
    "pl_rade", "pl_bmasse", "pl_dens", "pl_eqt", "pl_orbper", "sy_dist",
    "st_spectype", "st_lum", "st_teff", "st_met", "st_rad", "st_mass"
]

# Add missing optional columns (pl_dens, st_lum) if not present
for col in ["pl_dens", "st_lum"]:
    if col not in df.columns:
        df[col] = None

# Check which required columns are missing
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print("\n❌ Missing columns:", missing)
else:
    print("\n✔ All required columns exist!")

# -----------------------------
# 4️⃣ Save validated dataset
# -----------------------------
df.to_csv("data/exoplanets_validated.csv", index=False)
print("\n✔ Dataset saved as exoplanets_validated.csv")
