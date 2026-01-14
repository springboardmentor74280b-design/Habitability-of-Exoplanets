import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.pkl")

# -----------------------------
# Load model
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# Extract classifier safely
# -----------------------------
clf = model.steps[-1][1]   # 👈 last step of pipeline

# -----------------------------
# Feature names (USED IN TRAINING)
# -----------------------------
FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

# -----------------------------
# Get feature importance
# -----------------------------
importances = clf.feature_importances_

df_imp = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.barh(df_imp["Feature"], df_imp["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Habitability Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()

output_path = os.path.join(REPORTS_DIR, "feature_importance.png")
plt.savefig(output_path)
plt.close()

print("✅ Feature importance plot saved")
print("📁", output_path)
