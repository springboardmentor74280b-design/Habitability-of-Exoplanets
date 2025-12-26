import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale Features (CRITICAL for SVM)
# SVM calculates distances, so if one feature is 5000 (Temp) and another is 1 (Mass), SVM fails.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Weighted SVM
# class_weight='balanced' automatically sets higher penalties for Class 1 and 2
print("Training Weighted SVM...")
svm_model = SVC(
    kernel='rbf',               # Radial Basis Function (Good for non-linear islands)
    class_weight='balanced',    # <--- The Magic Command
    decision_function_shape='ovo', 
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = svm_model.predict(X_test_scaled)

print("\n--- Weighted SVM Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Weighted SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()