# ============================================
# MULTICLASS_IMBALANCED_ML_PIPELINE.PY
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, matthews_corrcoef
)

from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ============================================
# CONFIG
# ============================================
RANDOM_STATE = 42
N_SPLITS = 5
SAVE_DIR = "results/multiclass_pipeline"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================
data = pd.read_csv("cleaned_exoplanet_dataset.csv")

X = data.drop("P_HABITABLE", axis=1)
y = data["P_HABITABLE"]

# Keep numeric only (already cleaned)
X = X.select_dtypes(include=["int64", "float64"])

# ============================================
# TRAIN–TEST SPLIT (BEFORE SAMPLING)
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
# MODELS (MULTICLASS-CORRECT)
# ============================================
pipelines = {
    "ClassWeighted_SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(
            sampling_strategy="not majority",
            random_state=RANDOM_STATE
        )),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            decision_function_shape="ovr",
            random_state=RANDOM_STATE
        ))
    ]),

    "XGBoost": Pipeline([
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        ))
    ]),

    "Balanced_RF": Pipeline([
        ("model", BalancedRandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ))
    ])
}

# ============================================
# SCORING (MULTICLASS SAFE)
# ============================================
scoring = {
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "roc_auc_ovr": "roc_auc_ovr"
}

results = []

# ============================================
# TRAIN + EVALUATE
# ============================================
for name, pipe in pipelines.items():
    print(f"\nTraining {name}")

    cv = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        return_estimator=True,
        n_jobs=2
    )

    model = cv["estimator"][0]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Metrics
    roc_auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr"
    )



    

    mcc = matthews_corrcoef(y_test, y_pred)

###########
    

    # MUST be first
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt


###########

    from sklearn.metrics import f1_score, precision_score, recall_score

    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    results.append([
        name,
        roc_auc,
        mcc,
        f1,
        precision,
        recall
])

###########

   # results.append([name, roc_auc, mcc])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} – Confusion Matrix")
    plt.savefig(f"{SAVE_DIR}/{name}_confusion_matrix.png", dpi=300)
    plt.close()

# ============================================
# SAVE RESULTS
# ============================================

print("Results row length:", len(results[0]))

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "ROC_AUC_OVR",
        "MCC",
        "F1_macro",
        "Precision_macro",
        "Recall_macro"
    ]
)


results_df.to_csv(
    f"{SAVE_DIR}/evaluation_metrics.csv",
    index=False
)


###########


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "ROC_AUC_OVR",
        "MCC",
        "F1_macro",
        "Precision_macro",
        "Recall_macro"
    ]
)

results_df.to_csv(
    f"{SAVE_DIR}/evaluation_metrics.csv",
    index=False
)
###########
metrics_to_plot = ["F1_macro", "ROC_AUC_OVR", "MCC"]

results_df.set_index("Model")[metrics_to_plot].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Comparison on Imbalance-Aware Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/model_comparison_barplot.png", dpi=300)
plt.close()

###########

import seaborn as sns

plt.figure(figsize=(8, 5))
sns.heatmap(
    results_df.set_index("Model")[metrics_to_plot],
    annot=True,
    cmap="viridis",
    fmt=".3f"
)

plt.title("Performance Heatmap Across Models")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/metrics_heatmap.png", dpi=300)
plt.close()

###########

print("\nFinal Evaluation Metrics:")
print(results_df)
