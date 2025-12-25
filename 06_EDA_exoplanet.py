# =========================
# EXOPLANET EDA SCRIPT
# =========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
DATA_PATH = "scaled_exoplanet_dataset.csv"   # your scaled dataset
TARGET_COL = "P_HABITABLE"
SAVE_DIR = "EDA_plots"

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# =========================
# 1. CLASS IMBALANCE ANALYSIS
# =========================

plt.figure()
sns.countplot(x=y)
plt.title("Class Distribution: Habitability")
plt.xlabel("Habitability (0 = Not Habitable, 1 = Habitable)")
plt.ylabel("Count")
plt.savefig(f"{SAVE_DIR}/class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Percentage Distribution
class_percent = y.value_counts(normalize=True) * 100
print("\nClass Percentage Distribution:\n", class_percent)

# =========================
# 2. CORRELATION HEATMAP
# =========================

corr = X.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", linewidths=0.3)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{SAVE_DIR}/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 3. KDE PLOTS (TOP FEATURES)
# =========================

top_features = corr[TARGET_COL].abs().sort_values(ascending=False).index[:5] \
    if TARGET_COL in corr else X.columns[:5]

for col in top_features:
    plt.figure()
    sns.kdeplot(data=df, x=col, hue=TARGET_COL, fill=True)
    plt.title(f"KDE Distribution of {col}")
    plt.savefig(f"{SAVE_DIR}/kde_{col}.png", dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# 4. BOXPLOTS (OUTLIER DETECTION)
# =========================

for col in top_features:
    plt.figure()
    sns.boxplot(x=TARGET_COL, y=col, data=df)
    plt.title(f"Boxplot of {col} by Habitability")
    plt.savefig(f"{SAVE_DIR}/boxplot_{col}.png", dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# 5. SCATTER PLOTS (CLASS SEPARABILITY)
# =========================

if len(top_features) >= 2:
    plt.figure()
    sns.scatterplot(
        x=top_features[0],
        y=top_features[1],
        hue=TARGET_COL,
        data=df,
        alpha=0.7
    )
    plt.title(f"Scatter Plot: {top_features[0]} vs {top_features[1]}")
    plt.savefig(f"{SAVE_DIR}/scatter_features.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\nEDA completed. All plots saved in:", SAVE_DIR)
