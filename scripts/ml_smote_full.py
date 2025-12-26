# scripts/ml_smote_full.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -----------------------------
# 1️⃣ Load cleaned dataset
# -----------------------------
df = pd.read_csv("data/exoplanets_cleaned.csv")

# -----------------------------
# 2️⃣ Create habitability label
# -----------------------------
df["habitable"] = ((df["pl_rade"] < 2.5) &
                   (df["pl_eqt"] > 180) &
                   (df["pl_eqt"] < 400)).astype(int)

# Check class distribution
class_counts = df["habitable"].value_counts()
print("\nHabitability class distribution:\n", class_counts)

# If only one class exists, adjust thresholds dynamically
if class_counts.nunique() < 2:
    print("⚠ Only one class found. Adjusting thresholds based on dataset percentiles...")
    rade_thresh = df["pl_rade"].quantile(0.30)  # 30th percentile for radius
    eqt_min = df["pl_eqt"].quantile(0.10)      # 10th percentile for temperature
    eqt_max = df["pl_eqt"].quantile(0.90)      # 90th percentile for temperature

    df["habitable"] = ((df["pl_rade"] < rade_thresh) &
                       (df["pl_eqt"] > eqt_min) &
                       (df["pl_eqt"] < eqt_max)).astype(int)

    class_counts = df["habitable"].value_counts()
    print("\nAdjusted habitability class distribution:\n", class_counts)

# -----------------------------
# 3️⃣ Features & target
# -----------------------------
features = ["pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper",
            "pl_dens", "st_teff", "st_rad", "st_mass", "st_lum"]

X = df[features]
y = df["habitable"]

# -----------------------------
# 4️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 5️⃣ Conditional SMOTE
# -----------------------------
if y_train.nunique() > 1:
    smote = SMOTE(random_state=42)
    print("⚠ SMOTE enabled")
else:
    smote = None
    print("⚠ SMOTE disabled (only one class in training set)")

# -----------------------------
# 6️⃣ Define pipelines
# -----------------------------
pipelines = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ]) if smote is None else ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", smote),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ]),

    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ]) if smote is None else ImbPipeline([
        ("smote", smote),
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ]),

    "XGBoost": Pipeline([
        ("model", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                eval_metric="logloss", random_state=42))
    ]) if smote is None else ImbPipeline([
        ("smote", smote),
        ("model", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                eval_metric="logloss", random_state=42))
    ])
}

# -----------------------------
# 7️⃣ Train & evaluate all models
# -----------------------------
results = {}

for name, pipeline in pipelines.items():
    print(f"\n--- Training {name} ---")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} + {'SMOTE' if smote else 'No SMOTE'} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    results[name] = {
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# -----------------------------
# 8️⃣ Save the best-performing model (RandomForest)
# -----------------------------
joblib.dump(results["RandomForest"]["pipeline"], "artifacts/final_model.pkl")
print("\n✔ Final model saved to artifacts/final_model.pkl")
