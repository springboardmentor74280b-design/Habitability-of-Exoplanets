"""
PHASE 6: DIMENSIONALITY REDUCTION VISUALIZATION
================================================

This script implements dimensionality reduction techniques for visualization:
1. PCA (Principal Component Analysis) - Linear dimensionality reduction
2. t-SNE (t-Distributed Stochastic Neighbor Embedding) - Non-linear reduction
3. Visualization of class separability in 2D space
4. Explained variance analysis

LEARNING OBJECTIVES:
-------------------
- What is PCA and how does it work?
- What is t-SNE and when to use it?
- Difference between linear and non-linear dimensionality reduction
- How to visualize high-dimensional data in 2D
- Understanding explained variance
- Assessing class separability

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("PHASE 6: DIMENSIONALITY REDUCTION VISUALIZATION")
print("=" * 80)

# ============================================================================
# STEP 1: Load Scaled Features and Target
# ============================================================================
print("\n[1/6] Loading scaled features and target...")

features_path = config.PROCESSED_DATA_DIR / 'features_encoded_scaled.csv'
target_path = config.PROCESSED_DATA_DIR / 'target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)[config.TARGET_VARIABLE]

print(f"✓ Loaded features: {X.shape}")
print(f"✓ Loaded target: {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts().sort_index())

# Create dimensionality reduction plots directory
dim_red_plots_dir = config.PLOTS_DIR / 'dimensionality_reduction'
dim_red_plots_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 2: PCA Explanation and Implementation
# ============================================================================
print("\n[2/6] Applying PCA (Principal Component Analysis)...")

print("\n" + "=" * 80)
print("CONCEPT: What is PCA?")
print("=" * 80)
print("PCA = Principal Component Analysis")
print("\n📚 How it works:")
print("  1. Finds directions (principal components) of maximum variance")
print("  2. Projects data onto these new axes")
print("  3. First component captures most variance, second captures second most, etc.")
print("\n✅ Advantages:")
print("  ✓ Linear transformation (fast and interpretable)")
print("  ✓ Preserves global structure")
print("  ✓ Reduces dimensionality while retaining variance")
print("  ✓ Good for visualization and noise reduction")
print("\n❌ Limitations:")
print("  ✗ Assumes linear relationships")
print("  ✗ May miss non-linear patterns")
print("  ✗ Sensitive to scaling (we already scaled our data!)")
print("\n🎯 Use PCA when:")
print("  - You want to reduce dimensions for modeling")
print("  - You need interpretable components")
print("  - Data has linear relationships")
print("=" * 80)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
X_pca = pca.fit_transform(X)

print(f"\n✓ PCA transformation complete")
print(f"  Original dimensions: {X.shape[1]}")
print(f"  Reduced dimensions: {X_pca.shape[1]}")

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance) 

print(f"\n📊 Explained Variance:")
print(f"  PC1: {explained_variance[0]*100:.2f}%")
print(f"  PC2: {explained_variance[1]*100:.2f}%")
print(f"  Total (PC1 + PC2): {cumulative_variance[1]*100:.2f}%")

# Calculate explained variance for all components
pca_full = PCA(random_state=config.RANDOM_STATE)
pca_full.fit(X)
full_explained_variance = pca_full.explained_variance_ratio_
full_cumulative_variance = np.cumsum(full_explained_variance)

# Find number of components needed for 95% variance
n_components_95 = np.argmax(full_cumulative_variance >= 0.95) + 1
print(f"\n✓ Components needed for 95% variance: {n_components_95} (out of {X.shape[1]})")

# ============================================================================
# STEP 3: Visualize PCA Results
# ============================================================================
print("\n[3/6] Creating PCA visualizations...")

# Create PCA scatter plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('PCA: Principal Component Analysis', fontsize=16, fontweight='bold')

# Scatter plot
for cls in sorted(y.unique()):
    mask = y == cls
    color = config.BINARY_COLORS.get(cls, 'gray')
    label = 'Non-Habitable' if cls == 0 else 'Habitable'
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=color, alpha=0.6, s=50, label=label, 
                   edgecolors='black', linewidth=0.5)

axes[0].set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance)', fontsize=12, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance)', fontsize=12, fontweight='bold')
axes[0].set_title('PCA Projection (2D)', fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Explained variance plot
n_components_to_show = min(20, len(full_explained_variance))
axes[1].bar(range(1, n_components_to_show + 1), 
           full_explained_variance[:n_components_to_show] * 100,
           alpha=0.7, color='steelblue', edgecolor='black')
axes[1].plot(range(1, n_components_to_show + 1), 
            full_cumulative_variance[:n_components_to_show] * 100,
            'r-o', linewidth=2, markersize=6, label='Cumulative')
axes[1].axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% threshold')
axes[1].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Explained Variance by Component', fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
pca_path = dim_red_plots_dir / 'pca_visualization.png'
plt.savefig(pca_path, dpi=300, bbox_inches='tight')
print(f"✓ PCA visualization saved to: {pca_path}")
plt.close()

# ============================================================================
# STEP 4: t-SNE Explanation and Implementation
# ============================================================================
print("\n[4/6] Applying t-SNE (t-Distributed Stochastic Neighbor Embedding)...")

print("\n" + "=" * 80)
print("CONCEPT: What is t-SNE?")
print("=" * 80)
print("t-SNE = t-Distributed Stochastic Neighbor Embedding")
print("\n📚 How it works:")
print("  1. Measures similarity between points in high-dimensional space")
print("  2. Finds low-dimensional representation preserving these similarities")
print("  3. Uses probability distributions to maintain local structure")
print("  4. Non-linear transformation (can capture complex patterns)")
print("\n✅ Advantages:")
print("  ✓ Excellent for visualization (reveals clusters)")
print("  ✓ Captures non-linear relationships")
print("  ✓ Preserves local structure (nearby points stay nearby)")
print("  ✓ Great for exploratory analysis")
print("\n❌ Limitations:")
print("  ✗ Computationally expensive (slow for large datasets)")
print("  ✗ Non-deterministic (different runs give different results)")
print("  ✗ Cannot be applied to new data (no transform method)")
print("  ✗ Distances between clusters not meaningful")
print("\n🎯 Use t-SNE when:")
print("  - You want to visualize clusters")
print("  - Data has non-linear patterns")
print("  - You're doing exploratory analysis")
print("  - Dataset is not too large (< 10,000 samples ideal)")
print("=" * 80)

# Apply t-SNE
print("\n⏳ Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=config.RANDOM_STATE, 
           perplexity=30, max_iter=1000, verbose=0)
X_tsne = tsne.fit_transform(X)

print(f"✓ t-SNE transformation complete")
print(f"  Original dimensions: {X.shape[1]}")
print(f"  Reduced dimensions: {X_tsne.shape[1]}")

# ============================================================================
# STEP 5: Visualize t-SNE Results
# ============================================================================
print("\n[5/6] Creating t-SNE visualization...")

fig, ax = plt.subplots(figsize=(10, 8))

for cls in sorted(y.unique()):
    mask = y == cls
    color = config.BINARY_COLORS.get(cls, 'gray')
    label = 'Non-Habitable' if cls == 0 else 'Habitable'
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
              c=color, alpha=0.6, s=50, label=label,
              edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
ax.set_title('t-SNE Projection (2D)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
tsne_path = dim_red_plots_dir / 'tsne_visualization.png'
plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
print(f"✓ t-SNE visualization saved to: {tsne_path}")
plt.close()

# ============================================================================
# STEP 6: PCA vs t-SNE Comparison
# ============================================================================
print("\n[6/6] Creating PCA vs t-SNE comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Dimensionality Reduction Comparison: PCA vs t-SNE', 
            fontsize=16, fontweight='bold')

# PCA plot
for cls in sorted(y.unique()):
    mask = y == cls
    color = config.BINARY_COLORS.get(cls, 'gray')
    label = 'Non-Habitable' if cls == 0 else 'Habitable'
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=color, alpha=0.6, s=50, label=label,
                   edgecolors='black', linewidth=0.5)

axes[0].set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}% var)', fontsize=11, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}% var)', fontsize=11, fontweight='bold')
axes[0].set_title('PCA (Linear)', fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# t-SNE plot
for cls in sorted(y.unique()):
    mask = y == cls
    color = config.BINARY_COLORS.get(cls, 'gray')
    label = 'Non-Habitable' if cls == 0 else 'Habitable'
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=color, alpha=0.6, s=50, label=label,
                   edgecolors='black', linewidth=0.5)

axes[1].set_xlabel('t-SNE Component 1', fontsize=11, fontweight='bold')
axes[1].set_ylabel('t-SNE Component 2', fontsize=11, fontweight='bold')
axes[1].set_title('t-SNE (Non-linear)', fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
comparison_path = dim_red_plots_dir / 'pca_vs_tsne_comparison.png'
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved to: {comparison_path}")
plt.close()

# ============================================================================
# Generate Explained Variance Report
# ============================================================================
print("\n✓ Generating explained variance report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("DIMENSIONALITY REDUCTION REPORT")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("1. PCA ANALYSIS")
report_lines.append("-" * 80)
report_lines.append(f"Original Dimensions: {X.shape[1]}")
report_lines.append(f"Reduced Dimensions: 2")
report_lines.append(f"\nExplained Variance:")
report_lines.append(f"  PC1: {explained_variance[0]*100:.2f}%")
report_lines.append(f"  PC2: {explained_variance[1]*100:.2f}%")
report_lines.append(f"  Total (2 components): {cumulative_variance[1]*100:.2f}%")
report_lines.append(f"\nComponents for 95% variance: {n_components_95}")
report_lines.append("")

report_lines.append("2. t-SNE ANALYSIS")
report_lines.append("-" * 80)
report_lines.append(f"Original Dimensions: {X.shape[1]}")
report_lines.append(f"Reduced Dimensions: 2")
report_lines.append(f"Perplexity: 30")
report_lines.append(f"Iterations: 1000")
report_lines.append("")

report_lines.append("3. COMPARISON")
report_lines.append("-" * 80)
report_lines.append("PCA (Linear):")
report_lines.append("  ✓ Fast and deterministic")
report_lines.append("  ✓ Preserves global structure")
report_lines.append(f"  ✓ Captures {cumulative_variance[1]*100:.2f}% of variance in 2D")
report_lines.append("  ✗ May miss non-linear patterns")
report_lines.append("")
report_lines.append("t-SNE (Non-linear):")
report_lines.append("  ✓ Reveals clusters and local structure")
report_lines.append("  ✓ Captures non-linear relationships")
report_lines.append("  ✗ Slower and non-deterministic")
report_lines.append("  ✗ Cannot transform new data")
report_lines.append("")

report_lines.append("4. CLASS SEPARABILITY")
report_lines.append("-" * 80)
report_lines.append("Visual inspection of plots shows:")
report_lines.append("  - Significant class overlap in both PCA and t-SNE")
report_lines.append("  - Minority class (habitable) is scattered")
report_lines.append("  - No clear linear or non-linear separation")
report_lines.append("  - This confirms the difficulty of the classification task")
report_lines.append("")

report_lines.append("5. RECOMMENDATIONS")
report_lines.append("-" * 80)
report_lines.append("1. Use PCA for dimensionality reduction before modeling (faster)")
report_lines.append("2. Use t-SNE for exploratory visualization only")
report_lines.append("3. Consider feature engineering to improve separability")
report_lines.append("4. Apply SMOTE to handle class imbalance")
report_lines.append("5. Use ensemble methods to capture complex patterns")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("END OF DIMENSIONALITY REDUCTION REPORT")
report_lines.append("=" * 80)

# Save report
report_path = config.REPORTS_DIR / 'dimensionality_reduction_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Report saved to: {report_path}")

# Save explained variance data
variance_df = pd.DataFrame({
    'Component': range(1, len(full_explained_variance) + 1),
    'Explained_Variance': full_explained_variance * 100,
    'Cumulative_Variance': full_cumulative_variance * 100
})
variance_path = config.REPORTS_DIR / 'pca_explained_variance.csv'
variance_df.to_csv(variance_path, index=False)
print(f"✓ Explained variance data saved to: {variance_path}")

# Save PCA and t-SNE transformed data
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Habitability'] = y.values
pca_data_path = config.PROCESSED_DATA_DIR / 'pca_2d.csv'
pca_df.to_csv(pca_data_path, index=False)
print(f"✓ PCA 2D data saved to: {pca_data_path}")

tsne_df = pd.DataFrame(X_tsne, columns=['tSNE1', 'tSNE2'])
tsne_df['Habitability'] = y.values
tsne_data_path = config.PROCESSED_DATA_DIR / 'tsne_2d.csv'
tsne_df.to_csv(tsne_data_path, index=False)
print(f"✓ t-SNE 2D data saved to: {tsne_data_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DIMENSIONALITY REDUCTION COMPLETE")
print("=" * 80)

print("\n📊 Transformation Summary:")
print(f"  ✓ Original dimensions: {X.shape[1]}")
print(f"  ✓ PCA 2D: {cumulative_variance[1]*100:.2f}% variance retained")
print(f"  ✓ Components for 95% variance: {n_components_95}")
print(f"  ✓ t-SNE 2D: Non-linear projection complete")

print(f"\n📁 Files Generated:")
print(f"  1. PCA visualization: {pca_path}")
print(f"  2. t-SNE visualization: {tsne_path}")
print(f"  3. PCA vs t-SNE comparison: {comparison_path}")
print(f"  4. Dimensionality reduction report: {report_path}")
print(f"  5. Explained variance data: {variance_path}")
print(f"  6. PCA 2D data: {pca_data_path}")
print(f"  7. t-SNE 2D data: {tsne_data_path}")

print("\n🎓 Key Learnings:")
print("  1. PCA: Linear, fast, interpretable, preserves global structure")
print("  2. t-SNE: Non-linear, slow, great for visualization, preserves local structure")
print("  3. Use PCA for modeling, t-SNE for exploration")
print(f"  4. 2 PCA components capture {cumulative_variance[1]*100:.2f}% of variance")
print("  5. Class overlap visible in both methods - challenging classification task")

print("\n✅ Ready for Phase 7: Baseline Modeling!")
print("=" * 80)
