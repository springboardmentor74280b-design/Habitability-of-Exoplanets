import pandas as pd
import numpy as np
import pickle
import requests
import io

# 1. Define the NASA API Query
base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
query = """
select pl_name, pl_bmasse, pl_rade, pl_orbper, pl_orbsmax, pl_eqt, 
       st_teff, st_lum, st_rad, st_mass, st_met
from ps
where default_flag=1
"""
params = {
    "query": query,
    "format": "csv"
}

print("📡 Connecting to NASA Exoplanet Archive...")
response = requests.get(base_url, params=params)

if response.status_code != 200:
    print("❌ Failed to download data from NASA.")
    exit()

print("✅ Data Downloaded. Processing...")

# 2. Load into DataFrame
df_nasa = pd.read_csv(io.StringIO(response.text))

# Sanitize input columns
df_nasa.columns = df_nasa.columns.str.strip().str.lower()

# 3. Rename Columns
column_mapping = {
    'pl_name': 'P_NAME',
    'pl_bmasse': 'P_MASS_EST',
    'pl_rade': 'P_RADIUS_EST',
    'pl_orbper': 'P_PERIOD',
    'pl_orbsmax': 'P_DISTANCE',
    'pl_eqt': 'P_TEMP_EQUIL',
    'st_teff': 'S_TEMPERATURE',  # Maps to S_TEMPERATURE
    'st_lum': 'S_LUMINOSITY', 
    'st_rad': 'S_RADIUS',
    'st_mass': 'S_MASS',
    'st_met': 'S_METALLICITY'
}

df_nasa.rename(columns=column_mapping, inplace=True)

# 4. Data Cleaning
# NASA 'st_lum' conversion
if 'S_LUMINOSITY' in df_nasa.columns and df_nasa['S_LUMINOSITY'].max() < 100:
    df_nasa['S_LUMINOSITY'] = 10**df_nasa['S_LUMINOSITY']

# Filter: Use the EXACT columns your model knows
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]

# --- THE FIX: Smart Filtering ---
# Instead of dropping rows with *any* missing value, we only drop rows
# that are missing CRITICAL info (like Name).
# We trust the Pipeline's Imputer to fill in missing Physics data.
df_clean = df_nasa.dropna(subset=['P_NAME']).copy()

# Double Check: Ensure columns exist, fill missing cols with NaN if needed
for col in feature_cols:
    if col not in df_clean.columns:
        df_clean[col] = np.nan

print(f"🔬 Scannable Planets (Including those with missing data): {len(df_clean)}")

# 5. Load Your Champion AI
print("Loading SMOTE + XGBoost Model...")
with open('final_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# 6. Predict
# The pipeline will run 'SimpleImputer' automatically here!
X_nasa = df_clean[feature_cols]
predictions = model_pipeline.predict(X_nasa)
df_clean['AI_PREDICTION'] = predictions

# 7. Reveal the Discoveries
print("\n--- 🪐 AI DISCOVERY REPORT (NASA ARCHIVE) ---")

# Class 1: Conservative
class_1 = df_clean[df_clean['AI_PREDICTION'] == 1]
print(f"\n🏆 Class 1 (Conservative Candidates): {len(class_1)}")
print(class_1['P_NAME'].unique())

# Class 2: Optimistic
class_2 = df_clean[df_clean['AI_PREDICTION'] == 2]
print(f"\n✨ Class 2 (Optimistic Candidates): {len(class_2)}")
# Print first 30 names
print(class_2['P_NAME'].unique()[:30]) 

# Verification Check
print("\n--- 🕵️‍♂️ Verification Check ---")
famous_planets = ["TRAPPIST-1 e", "Proxima Cen b", "Kepler-186 f", "Teegarden's Star b", "LHS 1140 b"]

for planet in famous_planets:
    result = df_clean[df_clean['P_NAME'] == planet]
    if not result.empty:
        pred = result.iloc[0]['AI_PREDICTION']
        label = "Habitable (Class 1)" if pred == 1 else "Optimistic (Class 2)" if pred == 2 else "Non-Habitable"
        
        # Check what data was missing (for curiosity)
        missing = result[feature_cols].isnull().sum(axis=1).values[0]
        print(f"✅ {planet}: {label} (Had {missing} missing features)")
    else:
        print(f"⚠️ {planet}: Still not found in NASA database.")