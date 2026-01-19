# preprocessing.py
import pandas as pd
import numpy as np
import os

def load_and_clean_data(
    file_path=r"data\exoplanet_habitable.csv"
):
    df = pd.read_csv(file_path)

    # Keep only required columns
    df = df[[
        "P_NAME",
        "P_MASS",
        "P_RADIUS",
        "P_PERIOD",
        "P_TEMP_EQUIL",
        "S_TEMPERATURE",
        "S_METALLICITY",
        "S_TYPE",
        "P_HABITABLE"
    ]]

    df.rename(columns={
        "P_NAME": "pl_name",
        "P_MASS": "pl_bmasse",
        "P_RADIUS": "pl_rade",
        "P_PERIOD": "pl_orbper",
        "P_TEMP_EQUIL": "pl_eqt",
        "S_TEMPERATURE": "st_teff",
        "S_METALLICITY": "st_met",
        "S_TYPE": "st_spectype",
        "P_HABITABLE": "Habitability_raw"
    }, inplace=True)


    df["Habitability"] = (df["Habitability_raw"] >= 1).astype(int)
    df.drop(columns=["Habitability_raw"], inplace=True)

    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # One-hot encode spectral type
    df = pd.get_dummies(df, columns=["st_spectype"], drop_first=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/exoplanets_preprocessed.csv", index=False)

    print(" Preprocessing done")
    print("Habitability values:", df["Habitability"].value_counts())

    return df

if __name__ == "__main__":
    load_and_clean_data()
