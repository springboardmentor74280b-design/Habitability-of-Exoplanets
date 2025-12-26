# -----------------------------
# MODULE 3: Kernel PCA + KNN Classifier (Binary)
# -----------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

# -----------------------------
# 0. Create plots folder
# -----------------------------
os.makedirs("plots", exist_ok=True)

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("exoplanet_cleaned_final.csv")
print("Dataset loaded successfully")

# -----------------------------
# 2. Feature Selection based on Correlation
# -----------------------------
correlation = df.corr(numeric_only=True)['P.Habitable'].sort_values(ascending=False)
print("\nFeature correlation with Habitability:\n", correlation)

selected_features = ['P_RADIUS_EST', 'P_MASS_EST', 'P_SEMI_MAJOR_AXIS_EST']
X = df[selected_features]
y = df['P.Habitable']  # Binary: 1=Habitable, 0=Non-Habitable

# -----------------------------
# 3. Drop Missing Values
# -----------------------------
data = pd.concat([X, y], axis=1).dropna()
X = data[selected_features]
y = data['P.Habitable']
print(f"Final dataset size after NaN removal: {X.shape}")

# -----------------------------
# 4. Train-Test Split (80:20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Pipeline: Scaling + Kernel PCA + KNN Classifier
# -----------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', KernelPCA(n_components=2, kernel='rbf', random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# -----------------------------
# 6. Train Model
# -----------------------------
pipeline.fit(X_train, y_train)
print("Kernel PCA + KNN Classifier trained successfully")

# -----------------------------
# 7. Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# -----------------------------
# 8. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("KNN Classifier Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 9. ROC Curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'KNN Classifier (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("plots/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 10. Kernel PCA 2D Scatter Plot
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
X_kpca = kpca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_kpca[:,0], X_kpca[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.colorbar(label="Habitability Class")
plt.xlabel("Kernel PC 1")
plt.ylabel("Kernel PC 2")
plt.title("Kernel PCA 2D Projection of Exoplanets (Classification)")
plt.savefig("plots/kernel_pca_2d_class.png", dpi=300, bbox_inches="tight")
plt.show()
# -----------------------------
# 10A. t-SNE 2D Scatter Plot
# -----------------------------
from sklearn.manifold import TSNE

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.colorbar(label="Habitability Class")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE 2D Projection of Exoplanets")
plt.savefig("plots/tsne_2d_projection.png", dpi=300, bbox_inches="tight")
plt.show()

print("t-SNE plot saved successfully")


# -----------------------------
# 11. Habitability Class Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Distribution of Habitability Classes")
plt.savefig("plots/habitability_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 12. Feature Boxplot
# -----------------------------
plt.figure(figsize=(8,4))
sns.boxplot(data=X)
plt.title("Boxplot of Selected Features")
plt.savefig("plots/feature_boxplot_class.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nMODULE 3 (Classification) COMPLETED SUCCESSFULLY")
print("Plots saved in 'plots' folder")
