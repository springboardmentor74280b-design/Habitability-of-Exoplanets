import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(ARTIFACTS_DIR, "ranked_exoplanets.csv")

df = pd.read_csv(DATA_PATH)

top20 = df.head(20)

csv_path = os.path.join(REPORTS_DIR, "top_20_habitable_exoplanets.csv")
excel_path = os.path.join(REPORTS_DIR, "top_20_habitable_exoplanets.xlsx")

top20.to_csv(csv_path, index=False)
top20.to_excel(excel_path, index=False)

print("✅ Top candidates exported")
print(csv_path)
print(excel_path)
