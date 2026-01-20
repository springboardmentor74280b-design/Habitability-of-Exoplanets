import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer

# 1. Load Data
df = pd.read_csv('phl_exoplanet_catalog.csv')

feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]
target_col = 'P_HABITABLE'

# Clean & Prepare
df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(df_clean[feature_cols]), columns=feature_cols)
y = df_clean[target_col]

# 2. Split Data (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. COMPUTE WEIGHTS (The "Model-Level Technique")
# This creates a specific weight for every single row in X_train.
# Rare classes get huge weights, common classes get small weights.
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

print(f"Weights calculated. Example: Class 0 weight ~0.3, Class 1 weight ~20.0")

# 4. Train XGBoost with Weights
# We pass sample_weight to the fit() method
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

print("Training Weighted XGBoost...")
model.fit(X_train, y_train, sample_weight=sample_weights)

# 5. Evaluate
y_pred = model.predict(X_test)

print("\n--- XGBoost (Weighted) Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples')
plt.title("XGBoost Confusion Matrix\n(Weighted ")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# 6. Feature Importance (Required for Milestone 2)
# This answers: "What physical features drive habitability?"
plt.subplot(1, 2, 2)
xgb.plot_importance(model, importance_type='weight', max_num_features=10, ax=plt.gca())
plt.title("Feature Importance")

plt.tight_layout()
plt.show()