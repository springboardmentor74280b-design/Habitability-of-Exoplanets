# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv(
    r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS PROJECT DOC\cleaned_exoplanet_dataset.csv"
)

target_col = "P_HABITABLE"

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)


# =========================
# 3. ENCODE CATEGORICAL FEATURES
# =========================
cat_cols = X.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le


# =========================
# 4. TRAIN–TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Before SMOTE:")
print(y_train.value_counts())


# =========================
# 5. APPLY SMOTE (ONLY ON TRAINING DATA)
# =========================
smote = SMOTE(
    sampling_strategy="auto",
    k_neighbors=5,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# =========================
# 6. CREATE SMOTE FLAG (FOR ANALYSIS ONLY)
# =========================
n_original = len(X_train)
n_total = len(X_train_smote)

smote_flag = (
    ["original"] * n_original +
    ["synthetic"] * (n_total - n_original)
)

train_smote_full = X_train_smote.copy()
train_smote_full["smote_flag"] = smote_flag
train_smote_full["P_HABITABLE"] = y_train_smote

print("\nSMOTE flag counts:")
print(train_smote_full["smote_flag"].value_counts())

# ❗ DO NOT use smote_flag for training
X_train_final = X_train_smote.copy()


# =========================
# 7. SVM PIPELINE
# =========================
svm_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),

    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=10
    )),

    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    ))
])


# =========================
# 8. TRAIN MODEL
# =========================
svm_pipeline.fit(X_train_final, y_train_smote)


# =========================
# 9. PREDICTION & EVALUATION
# =========================
y_pred = svm_pipeline.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =========================
# 10. SELECTED FEATURES
# =========================
selected_mask = svm_pipeline.named_steps["feature_selection"].get_support()
selected_features = X_train.columns[selected_mask]

print("\nSelected Features:")
print(selected_features)
