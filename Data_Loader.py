# data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(
    filepath,
    target_col,
    test_size=0.2,
    random_state=42
):
    df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS_SPRINGBOARD_PROJECT\cleaned_exoplanet_dataset.csv")

    X = df.drop(columns=["P_HABITABLE"])
    y = df["P_HABITABLE"]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,       # 80:20 split
    random_state=42,
    stratify=y            # preserves class ratio
)

    return X_train, X_test, y_train, y_test
