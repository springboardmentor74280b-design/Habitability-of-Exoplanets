import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from model_utils import apply_physics_engine

# --- CONFIGURATION ---
DATA_FILE = 'phl_exoplanet_catalog.csv'
MODEL_FILE = 'exohab_model.joblib'
MATRIX_FILE = 'confusion_matrix.png'

print("⚙️ STARTING PRODUCTION TRAINING SEQUENCE...")

# 1. Load Data
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Raw Data Loaded ({len(df)} planets).")
except FileNotFoundError:
    print(f"❌ ERROR: '{DATA_FILE}' not found.")
    exit()

# 2. Apply Triple Physics Engine
print("⚛️ Applying Physics Engine (Mass <-> Radius <-> Temp)...")
df = apply_physics_engine(df)

# 3. Prepare Features
features = ['P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
            'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 
            'S_RADIUS', 'S_MASS', 'S_METALLICITY']
target = 'P_HABITABLE'

# Clean rows for training
df_clean = df.dropna(subset=[target, 'P_PERIOD'])
X = df_clean[features].fillna(-99)
y = df_clean[target]

print(f"📊 Dataset Ready: {len(X)} clean samples.")

# --- PHASE 1: VALIDATION (Generate Confusion Matrix) ---
print("\n--- PHASE 1: GENERATING METRICS (80/20 SPLIT) ---")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the Pipeline
def get_pipeline():
    return ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)), 
        ('xgb', xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=3, 
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])

# Train on 80%
val_pipeline = get_pipeline()
val_pipeline.fit(X_train, y_train)

# Predict on 20%
y_pred = val_pipeline.predict(X_test)

# Generate Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
# Dark Mode Style
plt.style.use('dark_background')
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Non-Habitable', 'Habitable', 'Optimistic'],
            yticklabels=['Non-Habitable', 'Habitable', 'Optimistic'])

plt.xlabel('AI Prediction', color='white')
plt.ylabel('Actual Label', color='white')
plt.title('ExoHab-AI Performance Matrix', color='#66fcf1', fontsize=14)
plt.savefig(MATRIX_FILE, transparent=False)
print(f"✅ Confusion Matrix saved to '{MATRIX_FILE}'")
print("📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Habitable', 'Habitable', 'Optimistic']))

# --- PHASE 2: FINAL PRODUCTION BUILD (Train on 100%) ---
print("\n--- PHASE 2: FINALIZING MODEL (100% DATA) ---")
print("🧠 Retraining AI on full dataset for maximum accuracy...")

final_pipeline = get_pipeline()
final_pipeline.fit(X, y) # <--- Training on EVERYTHING

# Save to Disk
joblib.dump(final_pipeline, MODEL_FILE)
print(f"💾 SUCCESS: Final Production Model saved to '{MODEL_FILE}'")
print("🚀 Ready for Deployment.")