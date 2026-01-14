import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "exoplanets_cleaned.csv")

FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

df = pd.read_csv(DATA_PATH)[FEATURES]

corr = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Planetary Features")

output = os.path.join(REPORTS_DIR, "feature_correlation_heatmap.png")
plt.tight_layout()
plt.savefig(output)
plt.close()

print("✅ Correlation heatmap saved")
print("📁", output)
