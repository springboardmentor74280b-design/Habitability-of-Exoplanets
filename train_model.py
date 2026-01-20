import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "exoplanet123.csv")

df = pd.read_csv(csv_path)

# Keep only numeric columns
df_numeric = df.select_dtypes(include=["int64", "float64"])

# Separate features and target
X = df_numeric.drop("P_HABITABLE", axis=1)
y = df_numeric["P_HABITABLE"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1_score": f1_score(y_test, y_pred, average="weighted")
}

print(metrics)

joblib.dump(model, os.path.join(BASE_DIR, "model", "model.pkl"))

with open(os.path.join(BASE_DIR, "model", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
