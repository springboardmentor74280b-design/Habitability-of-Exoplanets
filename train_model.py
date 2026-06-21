import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/exoplanet_dataset.csv")

x = df.drop("Habitable", axis=1)
y = df["Habitable"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)

pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "model/exoplanet_model.pkl")

print("Model Saved")
