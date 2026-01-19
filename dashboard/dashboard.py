import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def export_reports():
    df = pd.read_csv(os.path.join(DATA_DIR, "top_exoplanets.csv"))

    # Excel
    df.to_excel("Top_Habitable_Exoplanets.xlsx", index=False)

    print("Excel report exported")

if __name__ == "__main__":
    export_reports()
