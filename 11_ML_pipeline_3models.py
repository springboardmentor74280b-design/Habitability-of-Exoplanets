# ============================================
# FULL_IMBALANCED_ML_PIPELINE.PY
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, auc,
    matthews_corrcoef, ConfusionMatrixDisplay
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

# ============================================
# CONFIG
# ============================================
SAVE_DIR = "results/pipeline_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

RANDOM_STATE = 42
N_SPLITS = 5

# ============================================
# LOAD DATA
# ============================================
data = pd.read_csv("cleaned_exoplanet_dataset.csv")


X = data.drop("P_HABITABLE", axis=1)
y = data["P_HABITABLE"]

# categorical column is dropped

X = X.select_dtypes(include=["int64", "float64"])


# ============================================
# PREPROCESSING
# ============================================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ============================================
# PIPELINE
# ============================================
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", SVC(
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE
    ))
])

# ============================================
# TRAIN-TEST SPLIT (HOLD-OUT)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# ============================================
# STRATIFIED K-FOLD
# ============================================
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ============================================
# CROSS-VALIDATION (ONLY TRAIN DATA)
# ============================================
cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring=["f1", "roc_auc", "precision", "recall"],
    n_jobs=-1
)


# ============================================
# MODELS
# ============================================

pipelines = {
    "ClassWeighted_SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]),

    "XGBoost": Pipeline([
        ("model", XGBClassifier(
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ))
    ]),

    "Balanced_RF": Pipeline([
        ("model", BalancedRandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        ))
    ])
}

# ============================================
# SCORING METRICS
# ============================================
scoring = {
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "roc_auc_ovr": "roc_auc_ovr"
}


results = []

scoring=["f1", "roc_auc", "precision", "recall"]

results = []

# ============================================
# TRAIN + EVALUATE
# ============================================
for name, pipe in pipelines.items():
    print(f"\nTraining {name}")


    cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring=scoring,
    n_jobs=-1
)


    

    best_model = cv_results["estimator"][0]
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([name, roc, pr_auc, mcc])

    # ====================================
    # CONFUSION MATRIX
    # ====================================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{SAVE_DIR}/{name}_confusion_matrix.png", dpi=300)
    plt.close()

    # ====================================
    # PR CURVE
    # ====================================
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} Precision-Recall Curve")
    plt.savefig(f"{SAVE_DIR}/{name}_pr_curve.png", dpi=300)
    plt.close()

# ============================================
# SAVE METRICS
# ============================================
results_df = pd.DataFrame(
    results, columns=["Model", "ROC_AUC", "PR_AUC", "MCC"]
)
results_df.to_csv(f"{SAVE_DIR}/evaluation_metrics.csv", index=False)

print("\nFinal Evaluation Metrics:")
print(results_df)
