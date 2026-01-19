# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def add_features(
    input_file="data/exoplanets_preprocessed.csv",
    output_file="data/exoplanets_features.csv"
):
    if not os.path.exists(input_file):
        raise FileNotFoundError("Run preprocessing.py first")

    df = pd.read_csv(input_file)

    scaler = MinMaxScaler()

    features_to_norm = [
        "pl_eqt",
        "pl_rade",
        "pl_bmasse",
        "pl_orbper",
        "st_teff",
        "st_met"
    ]

    for col in features_to_norm:
        if col in df.columns:
            df[col + "_norm"] = scaler.fit_transform(df[[col]])

    # Habitability Score
    
    df["Habitability_Score"] = 0

    if "pl_eqt_norm" in df.columns:
        df["Habitability_Score"] += 1 - abs(df["pl_eqt_norm"] - 0.5)

    if "pl_rade_norm" in df.columns:
        df["Habitability_Score"] += 1 - abs(df["pl_rade_norm"] - 0.5)

    if "pl_bmasse_norm" in df.columns:
        df["Habitability_Score"] += 1 - abs(df["pl_bmasse_norm"] - 0.5)

    if "st_teff_norm" in df.columns:
        df["Habitability_Score"] += 1 - abs(df["st_teff_norm"] - 0.5)

 
    # Stellar Compatibility
    
    df["Stellar_Compatibility"] = 0

    if "st_teff_norm" in df.columns:
        df["Stellar_Compatibility"] += 1 - abs(df["st_teff_norm"] - 0.5)

    if "st_met_norm" in df.columns:
        df["Stellar_Compatibility"] += 1 - abs(df["st_met_norm"] - 0.5)

   
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_file, index=False)

    print(" Feature engineering completed")
    print("Shape:", df.shape)
    print("Habitability distribution:")
    print(df["Habitability"].value_counts())

    return df

if __name__ == "__main__":
    add_features()
