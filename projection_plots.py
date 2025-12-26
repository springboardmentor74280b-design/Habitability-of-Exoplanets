import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Load Original Data (to keep 0, 1, 2 labels)
try:
    df = pd.read_csv('phl_exoplanet_catalog.csv')
except FileNotFoundError:
    print("❌ Error: 'phl_exoplanet_catalog.csv' not found.")
    exit()

# Features
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]
target_col = 'P_HABITABLE'

# Clean & Impute
df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
imputer = SimpleImputer(strategy='median')
# Fill missing values
df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])

X = df_clean[feature_cols]
y = df_clean[target_col]

# Scale Features for PCA/t-SNE (Standardization is crucial here)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define Colors & Labels
colors = {0: 'red', 1: 'green', 2: 'blue'}
labels = {0: '0: Non-Hab', 1: '1: Conservative', 2: '2: Optimistic'}

# --- SETUP PLOTS (1 Row, 3 Columns) ---
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# --- PLOT 1: CORRELATION MATRIX (With Target) ---
# We combine X and y temporarily to see correlations with P_HABITABLE
combined_df = pd.DataFrame(X_scaled, columns=feature_cols)
combined_df['P_HABITABLE'] = y.values # Add target to matrix

corr_matrix = combined_df.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            ax=axes[0], cbar=False, annot_kws={"size": 8})
axes[0].set_title("Correlation Matrix (Includes P_HABITABLE)")

# --- PLOT 2: PCA (3 Classes) ---
print("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for class_val in [0, 2, 1]: # Order: 0(Red) -> 2(Blue) -> 1(Green) on top
    subset = X_pca[y == class_val]
    axes[1].scatter(subset[:, 0], subset[:, 1], c=colors[class_val], 
                    label=labels[class_val], alpha=0.6, s=25)

axes[1].set_title("PCA Projection")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# --- PLOT 3: t-SNE (3 Classes) ---
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

for class_val in [0, 2, 1]:
    subset = X_tsne[y == class_val]
    axes[2].scatter(subset[:, 0], subset[:, 1], c=colors[class_val], 
                    label=labels[class_val], alpha=0.6, s=25)

axes[2].set_title("t-SNE Projection")
axes[2].set_xlabel("Dim 1")
axes[2].set_ylabel("Dim 2")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ All plots generated.")