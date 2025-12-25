import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

from Data_Loader import load_and_split_data

# ---------------------------
# Load data
# ---------------------------
X_train, X_test, y_train, y_test = load_and_split_data(
    "cleaned_exoplanet_dataset.csv",
    target_col="P_HABITABLE"
)

# ---------------------------
# Keep numeric features
# ---------------------------
X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test = X_test.select_dtypes(include=["int64", "float64"])

# ---------------------------
# Apply SMOTE (TRAIN ONLY)
# ---------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", y_train_smote.value_counts())

# ---------------------------
# SAVE SMOTE DATASET (OPTIONAL)
# ---------------------------
smote_df = pd.concat(
    [
        pd.DataFrame(X_train_smote, columns=X_train.columns),
        pd.Series(y_train_smote, name="P_HABITABLE")
    ],
    axis=1
)

smote_df.to_csv("train_smote_exoplanet.csv", index=False)
print("SMOTE training dataset saved as train_smote_exoplanet.csv")

# ---------------------------
# Pipelines
# ---------------------------
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
])

nb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("nb", GaussianNB())
])

# ---------------------------
# Train & Evaluate
# ---------------------------
models = {
    "Logistic Regression": lr_pipeline,
    "KNN": knn_pipeline,
    "Naive Bayes": nb_pipeline
}

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    print(f"\n{name} (SMOTE):\n", classification_report(y_test, y_pred))
