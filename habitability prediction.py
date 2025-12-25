# ==========================================
# Exoplanet Habitability Prediction Project
# Module 4: Model Training, Evaluation & Plots
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

from xgboost import XGBClassifier

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
df = pd.read_csv("phl_exoplanet_catalog_2019.csv")

print("\nDataset Loaded Successfully")
print("Total records:", len(df))

# ------------------------------------------
# 2. Handle Column Names Safely
# ------------------------------------------
# Detect correct habitability column
if "P_HABITABLE" in df.columns:
    target_col = "P_HABITABLE"
elif "P.Habitable" in df.columns:
    target_col = "P.Habitable"
else:
    raise ValueError("Habitability column not found in dataset")

# Detect density column
if "P_DENSITY" in df.columns:
    density_col = "P_DENSITY"
elif "P_DENSITY_EST" in df.columns:
    density_col = "P_DENSITY_EST"
else:
    density_col = None  # optional feature

# ------------------------------------------
# 3. Feature Selection
# ------------------------------------------
features = [
    "P_RADIUS_EST",
    "P_MASS_EST",
    "P_SEMI_MAJOR_AXIS_EST",
    "P_TEMP_EQUIL"
]

if density_col:
    features.append(density_col)

# Drop missing values safely
df = df.dropna(subset=features + [target_col])

X = df[features]
y_binary = df[target_col]

print("\nFeatures used:", features)
print("Target column:", target_col)

# ------------------------------------------
# 4. Multi-Class Habitability Labels
# ------------------------------------------
def habitability_class(row):
    if row[target_col] == 1:
        return 2  # Highly Habitable
    elif row["P_TEMP_EQUIL"] < 250:
        return 1  # Potentially Habitable
    else:
        return 0  # Non-Habitable

df["Habitability_Level"] = df.apply(habitability_class, axis=1)
y_multi = df["Habitability_Level"]

# ------------------------------------------
# 5. Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

y_train_multi = y_multi.loc[y_train.index]
y_test_multi = y_multi.loc[y_test.index]

# ------------------------------------------
# 6. Random Forest (Binary)
# ------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest (Binary) ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))
# ------------------------------------------
# 6A. Actual vs Predicted Plot (Random Forest)
# ------------------------------------------
plt.figure(figsize=(6,4))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.xlabel("Actual Habitability")
plt.ylabel("Predicted Habitability")
plt.title("Actual vs Predicted Habitability (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.show()


# ------------------------------------------
# 7. Logistic Regression
# ------------------------------------------
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:, 1]

# ------------------------------------------
# 8. SVM
# ------------------------------------------
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True))
])

svm.fit(X_train, y_train)
svm_prob = svm.predict_proba(X_test)[:, 1]

# ------------------------------------------
# 9. ROC Curve Plot
# ------------------------------------------
plt.figure(figsize=(7,5))
for probs, label in [
    (rf_prob, "Random Forest"),
    (lr_prob, "Logistic Regression"),
    (svm_prob, "SVM")
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=label)

plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curve.png")
plt.show()

# ------------------------------------------
# 10. Confusion Matrix (RF)
# ------------------------------------------
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()

# ------------------------------------------
# 11. Feature Importance
# ------------------------------------------
importance = rf.feature_importances_

plt.figure(figsize=(7,4))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.show()

# ------------------------------------------
# 12. XGBoost (Multi-Class)
# ------------------------------------------
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train_multi)
xgb_pred = xgb.predict(X_test)

print("\n=== XGBoost (Multi-Class) ===")
print(classification_report(y_test_multi, xgb_pred))

cm_multi = confusion_matrix(y_test_multi, xgb_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm_multi, annot=True, fmt="d", cmap="Greens")
plt.title("XGBoost Multi-Class Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig("plots/xgb_multiclass_cm.png")
plt.show()

# ------------------------------------------
# 13. Rank Exoplanets
# ------------------------------------------
df["Habitability_Probability"] = rf.predict_proba(X)[:, 1]

top10 = df.sort_values(
    by="Habitability_Probability",
    ascending=False
)[["Habitability_Probability"]].head(10)

top10.to_csv("ranked_exoplanets.csv", index=False)

plt.figure(figsize=(8,4))
sns.barplot(
    x=top10["Habitability_Probability"],
    y=top10.index.astype(str)
)
plt.title("Top 10 Most Habitable Exoplanets")
plt.xlabel("Habitability Score")
plt.tight_layout()
plt.savefig("plots/top10_exoplanets.png")
plt.show()

print("\n Script executed successfully")
print(" Plots saved in /plots")
print(" Ranked exoplanets saved as ranked_exoplanets.csv")
