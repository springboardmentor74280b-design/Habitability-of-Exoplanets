"""
PHASE 12: MODEL INTERPRETABILITY
=================================

This script analyzes feature importance to understand
which features most influence habitability predictions.

LEARNING OBJECTIVES:
-------------------
- Understanding feature importance
- Model interpretability techniques
- Identifying key habitability factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

import config

sns.set_style('whitegrid')

print("=" * 80)
print("PHASE 12: MODEL INTERPRETABILITY")
print("=" * 80)

# Load data
print("\n[1/3] Loading data...")
features_path = config.PROCESSED_DATA_DIR / 'features_encoded_scaled.csv'
target_path = config.PROCESSED_DATA_DIR / 'target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)[config.TARGET_VARIABLE]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)

# Train Random Forest for feature importance (SVM doesn't have feature_importances_)
print("\n[2/3] Training Random Forest for interpretability...")
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=config.RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
rf_model.fit(X_train_resampled, y_train_resampled)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 Top {config.TOP_N_FEATURES} Most Important Features:")
for idx, row in feature_importance.head(config.TOP_N_FEATURES).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Visualize
print("\n[3/3] Creating feature importance visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top features
top_features = feature_importance.head(config.TOP_N_FEATURES)
axes[0].barh(range(len(top_features)), top_features['importance'].values, color='#2ecc71', alpha=0.7)
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels(top_features['feature'].values)
axes[0].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[0].set_title(f'Top {config.TOP_N_FEATURES} Most Important Features', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Cumulative importance
cumsum = feature_importance['importance'].cumsum()
axes[1].plot(range(len(cumsum)), cumsum, linewidth=2, color='#3498db')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
axes[1].set_xlabel('Number of Features', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
importance_viz_path = config.PLOTS_DIR / 'feature_importance.png'
plt.savefig(importance_viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Feature importance visualization saved to: {importance_viz_path}")
plt.close()

# Save feature importance
importance_path = config.REPORTS_DIR / 'feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"✓ Feature importance saved to: {importance_path}")

print("\n" + "=" * 80)
print("PHASE 12 COMPLETE: Model Interpretability")
print("=" * 80)
print(f"✅ Identified {config.TOP_N_FEATURES} most important features")
print(f"📁 Visualization: {importance_viz_path}")
print("=" * 80)
