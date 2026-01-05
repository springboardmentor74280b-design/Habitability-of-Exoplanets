# scripts/eda_pca_tsne.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Create output directory
# -----------------------------
OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load CLEANED data
# -----------------------------
df = pd.read_csv("data/exoplanets_cleaned.csv")
print("Loaded cleaned dataset:", df.shape)

# -----------------------------
# Create Habitability Label (FINAL LOGIC)
# -----------------------------
df["habitable"] = (
    (df["pl_rade"] < 2.5) &
    (df["pl_eqt"] > 180) &
    (df["pl_eqt"] < 400)
).astype(int)

# -----------------------------
# Feature Selection (FINAL FEATURES)
# -----------------------------
features = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "pl_dens"
]

X = df[features]
y = df["habitable"]

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA (2D)
# -----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=y,
    palette={0: "red", 1: "green"},
    alpha=0.8
)
plt.title("PCA Projection of Exoplanet Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Habitable")
plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "pca_plot.png")
plt.savefig(pca_path, dpi=300)
plt.close()
print(f"✔ PCA plot saved to {pca_path}")

# -----------------------------
# t-SNE (2D)
# -----------------------------
tsne = TSNE(
    n_components=2,
    perplexity=15,
    learning_rate=200,
    max_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=y,
    palette={0: "red", 1: "green"},
    alpha=0.8
)
plt.title("t-SNE Projection of Exoplanet Features")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title="Habitable")
plt.tight_layout()

tsne_path = os.path.join(OUTPUT_DIR, "tsne_plot.png")
plt.savefig(tsne_path, dpi=300)
plt.close()
print(f"✔ t-SNE plot saved to {tsne_path}")

print("\n✅ EDA PCA & t-SNE completed successfully")
