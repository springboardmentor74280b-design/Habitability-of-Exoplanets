


import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PATHS ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Make sure static folder exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Data and model paths
DATA_PATH = os.path.join(DATA_DIR, "exoplanet_habitable.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.joblib")

# Selected features
FEATURES = [
    "P_PERIOD","P_MASS","P_RADIUS","P_ECCENTRICITY","P_TEMP_EQUIL",
    "P_FLUX","S_TEMPERATURE","S_METALLICITY","S_RADIUS","S_MASS",
    "S_LOG_G","S_DISTANCE"
]

# ---------------- LOAD DATA & MODEL ---------------- #
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# ---------------- FEATURE IMPORTANCE ---------------- #
def feature_importance():
    plt.figure(figsize=(10,6))
    plt.barh(FEATURES, model.feature_importances_)
    plt.title("Feature Importance (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    save_path = os.path.join(STATIC_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(" Feature importance saved:", save_path)

# ---------------- HABITABILITY DISTRIBUTION ---------------- #

def habitability_distribution():
    plt.figure(figsize=(6,4))
    sns.countplot(x="P_HABITABLE", data=df, palette="viridis")
    plt.title("Habitability Class Distribution")
    plt.xlabel("Habitability")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = os.path.join(STATIC_DIR, "habitability_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(" Habitability distribution saved:", save_path)


# ---------------- CORRELATION HEATMAP ---------------- #
def correlation_heatmap():
    plt.figure(figsize=(12,10))
    corr = df[FEATURES].corr()
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Star–Planet Feature Correlation")
    plt.tight_layout()
    save_path = os.path.join(STATIC_DIR, "correlation_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(" Correlation heatmap saved:", save_path)

# ---------------- TOP 10 HABITABLE PLANETS ---------------- #
def top_habitable_planets():
    top_planets = df[df["P_HABITABLE"] == 1].copy()
    top_planets = top_planets.sort_values(by=["P_FLUX", "P_TEMP_EQUIL"], ascending=False)
    top10 = top_planets.head(10)
    
    # Save as table image
    fig, ax = plt.subplots(figsize=(12,3))
    ax.axis('tight')
    ax.axis('off')
    table_data = top10[["P_NAME","P_PERIOD","P_MASS","P_RADIUS","P_TEMP_EQUIL","P_FLUX"]]
    ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
    save_path = os.path.join(STATIC_DIR, "top10_planets.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(" Top 10 planets saved:", save_path)
    
    

# ---------------- RUN ALL ---------------- #
if __name__ == "__main__":
    feature_importance()
    habitability_distribution()
    correlation_heatmap()
    top_habitable_planets()
    print("\n All plots generated successfully in static/ folder")

