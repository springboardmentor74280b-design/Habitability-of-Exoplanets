import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ---------------------------
# Create EDA directory
# ---------------------------
os.makedirs("eda", exist_ok=True)

# ---------------------------
# Load datasets
# ---------------------------
df_smote = pd.read_csv("train_smote_exoplanet.csv")
df_original = pd.read_csv("cleaned_exoplanet_dataset.csv")

X_smote = df_smote.drop("P_HABITABLE", axis=1)
y_smote = df_smote["P_HABITABLE"]

X_orig = df_original.drop("P_HABITABLE", axis=1).select_dtypes(include=["int64", "float64"])
y_orig = df_original["P_HABITABLE"]

# =====================================================
# 1. CLASS DISTRIBUTION: BEFORE vs AFTER SMOTE
# =====================================================
fig, ax = plt.subplots(1, 2, figsize=(10,4))

y_orig.value_counts().plot(kind="bar", ax=ax[0], title="Before SMOTE")
y_smote.value_counts().plot(kind="bar", ax=ax[1], title="After SMOTE")

for a in ax:
    a.set_xlabel("Habitability Class")
    a.set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda/class_distribution_smote.png", dpi=300)
plt.close()

# =====================================================
# 2. PCA BEFORE vs AFTER SMOTE
# =====================================================
pca = PCA(n_components=2, random_state=42)

X_orig_pca = pca.fit_transform(X_orig)
X_smote_pca = pca.fit_transform(X_smote)

fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].scatter(
    X_orig_pca[:,0], X_orig_pca[:,1],
    c=y_orig, cmap="viridis", alpha=0.6
)
ax[0].set_title("PCA Before SMOTE")

ax[1].scatter(
    X_smote_pca[:,0], X_smote_pca[:,1],
    c=y_smote, cmap="viridis", alpha=0.6
)
ax[1].set_title("PCA After SMOTE")

for a in ax:
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")

plt.tight_layout()
plt.savefig("eda/pca_before_vs_after_smote.png", dpi=300)
plt.close()

# =====================================================
# 3. MINORITY CLASS DENSITY EXPANSION
# =====================================================
minority_class = y_smote.value_counts().idxmin()
top_features = X_smote.var().sort_values(ascending=False).head(3).index

plt.figure(figsize=(9,5))
for feature in top_features:
    sns.kdeplot(
        X_smote[y_smote == minority_class][feature],
        label=feature,
        fill=True,
        alpha=0.4
    )

plt.title("Minority Class Feature Density After SMOTE")
plt.xlabel("Feature Value")
plt.legend()
plt.tight_layout()
plt.savefig("eda/minority_density_smote.png", dpi=300)
plt.close()

# =====================================================
# 4. CORRELATION HEATMAP (AFTER SMOTE)
# =====================================================
plt.figure(figsize=(10,8))
sns.heatmap(
    X_smote.corr(),
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Feature Correlation After SMOTE")
plt.tight_layout()
plt.savefig("eda/correlation_after_smote.png", dpi=300)
plt.close()

print("✅ All SMOTE EDA plots saved in /eda folder")
