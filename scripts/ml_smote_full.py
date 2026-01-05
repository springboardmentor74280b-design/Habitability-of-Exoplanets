# scripts/ml_smote_full.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv("data/exoplanets_cleaned.csv")

# -----------------------------
# Create habitability label
# -----------------------------
df["habitable"] = (
    (df["pl_rade"] < 2.5) &
    (df["pl_eqt"] > 180) &
    (df["pl_eqt"] < 400)
).astype(int)

print("\nClass distribution:")
print(df["habitable"].value_counts())

# -----------------------------
# Features & target
# -----------------------------
features = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

X = df[features]
y = df["habitable"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------------
# Model with SMOTE
# -----------------------------
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(pipeline, "artifacts/final_model.pkl")
print("\n✔ Model saved to artifacts/final_model.pkl")
