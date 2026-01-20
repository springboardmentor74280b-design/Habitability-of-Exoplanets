import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# 1. Load Data
try:
    df = pd.read_csv('phl_exoplanet_catalog.csv')
except FileNotFoundError:
    print("❌ Error: 'phl_exoplanet_catalog.csv' not found.")
    exit()

# Features
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]
target_col = 'P_HABITABLE'

# 2. Clean & Prepare
# Drop rows where target is missing
df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])

# Impute missing feature values with Median
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df_clean[feature_cols])
y = df_clean[target_col]

print(f"Class Distribution:\n{y.value_counts()}")

# 3. Split & Scale
# We use 'stratify=y' to ensure test set has some class 1 and 2 planets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (Critical for SVM and LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define Baseline Models
# We use class_weight='balanced' to help them fight the imbalance slightly
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
    "Linear SVM": SVC(kernel='linear', class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
}

# 5. Train & Evaluate
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    print(f"\n--- Training {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Text Report
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
    axes[i].set_xticklabels(['0', '1', '2'])
    axes[i].set_yticklabels(['0', '1', '2'])

plt.tight_layout()
plt.show()

print("\n✅ Baseline Training Complete.")