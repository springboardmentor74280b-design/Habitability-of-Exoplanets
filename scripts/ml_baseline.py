import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --------------------------------------------------
# 1️⃣ Load RAW data (for labeling)
# --------------------------------------------------
df_raw = pd.read_csv("data/exoplanets_validated.csv")

# --------------------------------------------------
# 2️⃣ Create Habitability Label (RELAXED BASELINE HEURISTIC)
# --------------------------------------------------
df_raw["habitable"] = (
    (df_raw["pl_rade"] < 2.5) &              # relaxed radius
    (df_raw["pl_eqt"] > 180) &               # relaxed temperature
    (df_raw["pl_eqt"] < 400)
).astype(int)

print("\nHabitability Distribution (RAW):")
print(df_raw["habitable"].value_counts())

# 🚨 Safety check (VERY IMPORTANT)
if df_raw["habitable"].nunique() < 2:
    raise ValueError(
        "❌ Only one class detected. Relax habitability thresholds further."
    )

# --------------------------------------------------
# 3️⃣ Load CLEANED data (for ML)
# --------------------------------------------------
df = pd.read_csv("data/exoplanets_cleaned.csv")
print("\nDataset Loaded:", df.shape)

# Attach label to cleaned dataset
df["habitable"] = df_raw["habitable"].values

# --------------------------------------------------
# 4️⃣ Select Features
# --------------------------------------------------
features = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_lum"
]

X = df[features]
y = df["habitable"]

# --------------------------------------------------
# 5️⃣ Train-Test Split (STRATIFIED)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# --------------------------------------------------
# 6️⃣ BASELINE MODEL 1: Logistic Regression
# --------------------------------------------------
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
])

logreg_pipeline.fit(X_train, y_train)

y_pred_lr = logreg_pipeline.predict(X_test)
y_prob_lr = logreg_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression ---")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

# --------------------------------------------------
# 7️⃣ BASELINE MODEL 2: Random Forest
# --------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# --------------------------------------------------
# 8️⃣ BASELINE MODEL 3: XGBoost
# --------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("\n--- XGBoost ---")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
