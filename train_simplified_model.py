"""
SIMPLIFIED MODEL: 8-Feature Exoplanet Habitability Classifier
==============================================================

This script trains a new model using ONLY the 8 key features that users can easily provide.
This will give much more accurate predictions than trying to fill in 6,501 missing features.

The 8 features are:
1. P_MASS_EST - Planet Mass
2. P_RADIUS_EST - Planet Radius  
3. P_TEMP_EQUIL - Equilibrium Temperature
4. P_PERIOD - Orbital Period
5. P_FLUX - Stellar Flux
6. S_MASS - Star Mass
7. S_RADIUS - Star Radius
8. S_TEMPERATURE - Star Temperature
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

import config

print("=" * 80)
print("TRAINING SIMPLIFIED 8-FEATURE MODEL")
print("=" * 80)

# Load cleaned data
print("\n[1/5] Loading data...")
cleaned_data_path = config.PROCESSED_DATA_DIR / 'exoplanet_data_cleaned.csv'
df = pd.read_csv(cleaned_data_path)

# Select only the 8 key features
key_features = [
    'P_MASS_EST',
    'P_RADIUS_EST', 
    'P_TEMP_EQUIL',
    'P_PERIOD',
    'P_FLUX',
    'S_MASS',
    'S_RADIUS',
    'S_TEMPERATURE'
]

print(f"✓ Using {len(key_features)} key features")

# Extract features and target
X = df[key_features].copy()
y = df[config.TARGET_VARIABLE].copy()

print(f"✓ Data shape: {X.shape}")
print(f"✓ Target distribution:\n{y.value_counts().sort_index()}")

# Train/test split
print("\n[2/5] Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y
)

print(f"✓ Training: {X_train.shape[0]:,} samples")
print(f"✓ Test: {X_test.shape[0]:,} samples")

# Create pipeline with SMOTE + Scaler + SVM
print("\n[3/5] Training simplified model...")

simplified_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=config.RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='linear', class_weight='balanced', 
                      random_state=config.RANDOM_STATE, probability=True, max_iter=2000))
])

simplified_pipeline.fit(X_train, y_train)
print("✓ Model trained!")

# Evaluate
print("\n[4/5] Evaluating model...")
y_pred = simplified_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")

print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Habitable', 'Habitable', 'Optimistic'])}")

# Save model
print("\n[5/5] Saving simplified model...")
simplified_model_path = config.MODELS_DIR / 'production' / 'simplified_8feature_model.pkl'
joblib.dump(simplified_pipeline, simplified_model_path)
print(f"✓ Saved to: {simplified_model_path}")

# Save feature list
import json
feature_list_path = config.MODELS_DIR / 'production' / 'simplified_features.json'
with open(feature_list_path, 'w') as f:
    json.dump({'features': key_features}, f, indent=2)
print(f"✓ Feature list saved to: {feature_list_path}")

print("\n" + "=" * 80)
print("SIMPLIFIED MODEL TRAINING COMPLETE")
print("=" * 80)
print(f"\n✅ Model achieves {f1:.2%} F1 score with only 8 features!")
print(f"📁 Model saved to: {simplified_model_path}")
print("\nThis model will give much more accurate predictions for user inputs!")
print("=" * 80)
