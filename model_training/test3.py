import pandas as pd
import numpy as np
import pickle

# 1. Load the Uploaded File
filename = 'PSCompPars_2025.12.24_23.44.10.csv'
print(f"📂 Loading {filename}...")

# Note: The file has comments starting with '#', so we skip them automatically
df_nasa = pd.read_csv(filename, comment='#')

# Sanitize column names (lowercase, strip spaces)
df_nasa.columns = df_nasa.columns.str.strip().str.lower()

# 2. Rename Columns to Match Your AI's "Brain"
# Mapping NASA Composite Table -> Your Model Features
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

# 3. Data Cleaning & Unit Conversion
# NASA 'st_lum' is log(Solar) if values are small. We convert to linear Solar Units.
if 'S_LUMINOSITY' in df_nasa.columns:
    # If max value is small (< 100), assume it's Log10 and convert
    if df_nasa['S_LUMINOSITY'].max() < 100:
        print("⚠️ Detected Log-Luminosity. Converting to Solar Units...")
        df_nasa['S_LUMINOSITY'] = 10**df_nasa['S_LUMINOSITY']

# Define the exact features your model expects
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]

# 4. Filter Data
# We act leniently: we keep rows as long as they have a Name.
# We trust the Pipeline's Imputer to fill in missing physics.
df_clean = df_nasa.dropna(subset=['P_NAME']).copy()

# Ensure all columns exist (fill with NaN if missing, so Imputer handles them)
for col in feature_cols:
    if col not in df_clean.columns:
        df_clean[col] = np.nan

print(f"🔬 Scannable Planets: {len(df_clean)}")

# 5. Load Your Champion AI
print("🧠 Loading SMOTE + XGBoost Model...")
try:
    with open('final_pipeline.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: 'final_pipeline.pkl' not found. Make sure you are in the right folder.")
    exit()

# 6. Predict
X_nasa = df_clean[feature_cols]
predictions = model_pipeline.predict(X_nasa)
df_clean['AI_PREDICTION'] = predictions

# 7. Reveal the Discoveries
print("\n--- 🪐 AI DISCOVERY REPORT (NASA COMPOSITE DATA) ---")

# Class 1: Conservative
class_1 = df_clean[df_clean['AI_PREDICTION'] == 1]
print(f"\n🏆 Class 1 (Conservative Candidates): {len(class_1)}")
print(class_1['P_NAME'].unique())

# Class 2: Optimistic
class_2 = df_clean[df_clean['AI_PREDICTION'] == 2]
print(f"\n✨ Class 2 (Optimistic Candidates): {len(class_2)}")
# Print first 20 to keep it clean
print(class_2['P_NAME'].unique()[:20]) 

# 8. The "Famous Five" Verification Check
print("\n--- 🕵️‍♂️ Famous Planet Verification ---")
famous_planets = ["TRAPPIST-1 e", "Proxima Cen b", "Kepler-186 f", "Teegarden's Star b", "LHS 1140 b", "Kepler-452 b"]

for planet in famous_planets:
    result = df_clean[df_clean['P_NAME'] == planet]
    if not result.empty:
        pred = result.iloc[0]['AI_PREDICTION']
        
        # Translate code to text
        if pred == 1: label = "🏆 Habitable (Class 1)"
        elif pred == 2: label = "✨ Optimistic (Class 2)" 
        else: label = "💀 Non-Habitable"
        
        # Check for missing data (just for curiosity)
        missing = result[feature_cols].isnull().sum(axis=1).values[0]
        
        print(f"✅ {planet}: {label} (Missing info: {missing} fields)")
    else:
        print(f"⚠️ {planet}: Not found in file.")