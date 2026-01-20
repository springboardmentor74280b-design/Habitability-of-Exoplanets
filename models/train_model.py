import os
import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "exoplanet_habitable.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_OUT = os.path.join(BASE_DIR, "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)



df = pd.read_csv(DATA_PATH)


FEATURES = [
    "P_PERIOD",
    "P_MASS",
    "P_RADIUS",
    "P_ECCENTRICITY",
    "P_TEMP_EQUIL",
    "P_FLUX",
    "S_TEMPERATURE",
    "S_METALLICITY",
    "S_RADIUS",
    "S_MASS",
    "S_LOG_G",
    "S_DISTANCE"
]

TARGET = "P_HABITABLE"

df = df[FEATURES + [TARGET]].dropna()

X = df[FEATURES]
y = df[TARGET].apply(lambda x: 1 if x in [1, 2] else 0)
 

print("Class distribution:\n", y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))



smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("After SMOTE:\n", pd.Series(y_train_bal).value_counts())



model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42
)

model.fit(X_train_bal, y_train_bal)

joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.joblib"))



y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1] 

accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n RESULTS ")
print("Accuracy:", accuracy)
print("ROC AUC:", roc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Non-Habitable", "Habitable"],
    yticklabels=["Non-Habitable", "Habitable"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(" Confusion matrix saved:", cm_path)


# ================= OTHER MODELS CONFUSION MATRICES ================= #

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

for name, clf in models.items():
    clf.fit(X_train_bal, y_train_bal)
    y_pred_model = clf.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred_model)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Habitable", "Habitable"],
        yticklabels=["Non-Habitable", "Habitable"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    save_path = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}_cm.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f" {name} confusion matrix saved:", save_path)




# ---------------- PCA PLOT ---------------- #
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)  # use all X

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_full)

pca_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Habitability": y
})

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Habitability",
    palette={0: "red", 1: "green"},
    alpha=0.6
)
plt.title("PCA Projection of Exoplanets")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title="Habitability")
plt.tight_layout()
pca_path = os.path.join(MODEL_DIR, "pca_plot.png")
plt.savefig(pca_path, dpi=150)
plt.close()
print(" PCA plot saved:", pca_path)



pd.DataFrame({"feature": FEATURES}).to_csv(
    os.path.join(DATA_OUT, "selected_features.csv"),
    index=False
)


df_full = pd.read_csv(DATA_PATH)

top_planets = df_full[df_full["P_HABITABLE"] == 1].copy()

top_planets = top_planets.sort_values(
    by=["P_ESI", "P_FLUX"],
    ascending=False
)

TOP_COLS = [
    "P_NAME",
    "P_PERIOD",
    "P_MASS",
    "P_RADIUS",
    "P_TEMP_EQUIL",
    "P_FLUX",
    "S_NAME",
    "S_TEMPERATURE",
    "P_ESI"
]

top_planets[TOP_COLS].head(10).to_csv(
    os.path.join(DATA_OUT, "top_exoplanets.csv"),
    index=False
)

print("\n Training complete")
print(" xgboost_model.joblib saved")
print(" scaler.joblib saved")
print(" selected_features.csv saved")
print(" top_exoplanets.csv saved")
