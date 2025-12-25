import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ---------------------------
# Load SMOTE dataset
# ---------------------------
df = pd.read_csv("train_smote_exoplanet.csv")

X = df.drop("P_HABITABLE", axis=1)
y = df["P_HABITABLE"]

# ---------------------------
# 1. Class Distribution (After SMOTE)
# ---------------------------
plt.figure(figsize=(5,4))
y.value_counts().plot(kind="bar")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Habitability Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------------------
# 2. PCA Visualization
# ---------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6,5))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap="viridis",
    alpha=0.6
)
plt.title("PCA Projection of SMOTE Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Habitability")
plt.tight_layout()
plt.show()

# ---------------------------
# 3. Feature Distribution Shift
# ---------------------------
top_features = X.var().sort_values(ascending=False).head(5).index

plt.figure(figsize=(10,5))
for feature in top_features:
    sns.kdeplot(X[feature], label=feature)

plt.title("Top Feature Distributions After SMOTE")
plt.xlabel("Feature Value")
plt.legend()
plt.tight_layout()
plt.show()
