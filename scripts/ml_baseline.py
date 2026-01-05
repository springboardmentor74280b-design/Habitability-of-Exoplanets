# scripts/ml_baseline.py
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --------------------------------------------------
# 1️⃣ Load CLEANED data
# --------------------------------------------------
df = pd.read_csv("data/exoplanets_cleaned.csv")
print("Loaded cleaned dataset:", df.shape)

# --------------------------------------------------
# 2️⃣ Create Habitability Label (FINAL LOGIC)
# --------------------------------------------------
df["habitable"] = (
    (df["pl_rade"] < 2.5) &
    (df["pl_eqt"] > 180) &
    (df["pl_eqt"] < 400)
).astype(int)

print("\nHabitability Distribution:")
print(df["habitable"].value_counts())

if df["habitable"].nunique() < 2:
    raise ValueError("❌ Only one class found. Cannot train baseline models.")

# --------------------------------------------------
# 3️⃣ Feature Selection (FINAL FEATURES)
# --------------------------------------------------
features = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

X = df[features]
y = df["habitable"]

# --------------------------------------------------
# 4️⃣ Train-Test Split (STRATIFIED)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# 5️⃣ Logistic Regression (BASELINE)
# --------------------------------------------------
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

lr_pipeline.fit(X_train, y_train)

y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression (Baseline) ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

# --------------------------------------------------
# 6️⃣ Random Forest (BASELINE)
# --------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest (Baseline) ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
