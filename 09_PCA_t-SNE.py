import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


SAVE_DIR = "EDA_plots/dimensionality_reduction"
os.makedirs(SAVE_DIR, exist_ok=True)

#load dataset
df = pd.read_csv("cleaned_exoplanet_dataset.csv")

X = df.drop(columns=["P_HABITABLE"])
y = df["P_HABITABLE"]

X = pd.get_dummies(X, drop_first=True)


#Train–Test Split (IMPORTANT for accuracy)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


#PCA (2D) – Global Class Structure
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)

#PCA scatter plt
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[y_train == 0, 0],
    X_pca[y_train == 0, 1],
    c="red",
    alpha=0.6,
    label="Non-Habitable (0)"
)

plt.scatter(
    X_pca[y_train == 1, 0],
    X_pca[y_train == 1, 1],
    c="green",
    alpha=0.8,
    label="Habitable (1)"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA (2D) – Class Imbalance Visualization")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(SAVE_DIR, "pca_2d_class_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


#t-SNE – Minority Class Separability
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,   # ✅ changed from n_iter
    random_state=42
)

X_tsne = tsne.fit_transform(X_train_scaled)


#t-SNE scatter plt
plt.figure(figsize=(8, 6))

plt.scatter(
    X_tsne[y_train == 0, 0],
    X_tsne[y_train == 0, 1],
    c="red",
    alpha=0.6,
    label="Non-Habitable (0)"
)

plt.scatter(
    X_tsne[y_train == 1, 0],
    X_tsne[y_train == 1, 1],
    c="green",
    alpha=0.8,
    label="Habitable (1)"
)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE – Minority Class Separability")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(SAVE_DIR, "tsne_2d_minority_separability.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
