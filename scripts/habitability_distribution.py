import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(ARTIFACTS_DIR, "ranked_exoplanets.csv")

df = pd.read_csv(DATA_PATH)

plt.figure(figsize=(8, 5))
sns.histplot(df["habitability_score"], bins=30, kde=True)
plt.xlabel("Habitability Score")
plt.ylabel("Planet Count")
plt.title("Distribution of Habitability Scores")

output = os.path.join(REPORTS_DIR, "habitability_distribution.png")
plt.tight_layout()
plt.savefig(output)
plt.close()

print("✅ Habitability score distribution saved")
print("📁", output)
