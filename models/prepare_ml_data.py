import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def prepare_ml_dataset():
    df = pd.read_csv("data/exoplanets_features.csv")

    y = df["Habitability"]
    X = df.drop(columns=["pl_name", "Habitability"], errors="ignore")

    # Save feature list
    X.columns.to_series().to_csv("data/selected_features.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Balance training data
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train_scaled)
    np.save("data/X_test.npy", X_test_scaled)
    np.save("data/y_train.npy", y_train_bal)
    np.save("data/y_test.npy", y_test.values)
    # Save scaler
    import joblib
    joblib.dump(scaler, "models/scaler.joblib")

    return X_train_scaled, X_test_scaled, y_train_bal, y_test




if __name__ == "__main__":
    prepare_ml_dataset()
