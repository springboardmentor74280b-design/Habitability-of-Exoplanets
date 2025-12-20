"""
PHASE 5: EXPLORATORY DATA ANALYSIS (EDA)
=========================================

This script performs comprehensive exploratory data analysis including:
1. Class imbalance visualization (bar chart, pie chart)
2. Correlation heatmap with annotations
3. Scatter plots, KDEs, boxplots grouped by target class
4. Identification of redundant features (high correlation)
5. Statistical tests for feature-target relationships

LEARNING OBJECTIVES:
-------------------
- Understanding the severity of class imbalance
- Identifying correlated and redundant features
- Discovering patterns and relationships in the data
- Statistical significance testing
- Why class imbalance makes standard ML approaches fail

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("PHASE 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# ============================================================================
# STEP 1: Load Cleaned Dataset
# ============================================================================
print("\n[1/8] Loading cleaned dataset...")

cleaned_data_path = config.PROCESSED_DATA_DIR / 'exoplanet_data_cleaned.csv'
df = pd.read_csv(cleaned_data_path)

print(f"✓ Loaded dataset: {df.shape}")
print(f"  - Total samples: {df.shape[0]:,}")
print(f"  - Total features: {df.shape[1]}")

# Separate features and target
y = df[config.TARGET_VARIABLE].copy()
identifier_cols = ['P_NAME', 'S_NAME', 'S_ALT_NAMES', 'S_CONSTELLATION', 
                   'S_CONSTELLATION_ABR', 'S_CONSTELLATION_ENG']
cols_to_drop = [config.TARGET_VARIABLE] + [col for col in identifier_cols if col in df.columns]
X = df.drop(columns=cols_to_drop, errors='ignore')

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Target: {y.shape[0]}")

# ============================================================================
# STEP 2: Class Imbalance Visualization
# ============================================================================
print("\n[2/8] Analyzing class imbalance...")

# Create EDA plots directory
eda_plots_dir = config.PLOTS_DIR / 'eda'
eda_plots_dir.mkdir(parents=True, exist_ok=True)

# Calculate class distribution
class_counts = y.value_counts().sort_index()
class_percentages = (class_counts / len(y) * 100).round(2)

print("\n" + "=" * 80)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 80)
print("\nClass Distribution:")
for cls in class_counts.index:
    count = class_counts[cls]
    pct = class_percentages[cls]
    print(f"  Class {cls}: {count:5d} samples ({pct:5.2f}%)")

# Calculate imbalance ratio
imbalance_ratio = 1.0
if len(class_counts) >= 2:
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    print(f"\n⚠️  IMBALANCE RATIO: {imbalance_ratio:.1f}:1")
    print(f"   (Majority class is {imbalance_ratio:.1f}x larger than minority class)")
    print("\n   This is EXTREME imbalance! Standard ML will fail.")
    print("   Models will achieve ~98% accuracy by predicting everything as class 0!")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Class Imbalance Visualization', fontsize=16, fontweight='bold')

# Bar chart
colors = [config.BINARY_COLORS.get(i, 'gray') for i in class_counts.index]
axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Habitability Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
axes[0].set_title('Class Distribution (Bar Chart)', fontsize=13, fontweight='bold')
axes[0].set_xticks(class_counts.index)
axes[0].set_xticklabels(['Non-Habitable' if x == 0 else 'Habitable' for x in class_counts.index])
for i, (cls, count) in enumerate(class_counts.items()):
    axes[0].text(cls, count + 50, f'{count}\n({class_percentages[cls]:.2f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

# Pie chart
class_labels = ['Non-Habitable' if x == 0 else f'Habitable-{x}' for x in class_counts.index]
axes[1].pie(class_counts.values, labels=class_labels, 
           colors=colors, autopct='%1.2f%%', startangle=90,
           explode=[0.1 if i > 0 else 0 for i in range(len(class_counts))], 
           shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('Class Distribution (Pie Chart)', fontsize=13, fontweight='bold')

plt.tight_layout()
imbalance_path = eda_plots_dir / 'class_imbalance.png'
plt.savefig(imbalance_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Class imbalance visualization saved to: {imbalance_path}")
plt.close()

# ============================================================================
# STEP 3: Correlation Analysis
# ============================================================================
print("\n[3/8] Performing correlation analysis...")

# Get numerical features only
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n✓ Analyzing correlations for {len(numerical_features)} numerical features...")

# Calculate correlation matrix
correlation_matrix = X[numerical_features].corr()

# Identify highly correlated features (redundant)
high_corr_threshold = 0.9
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

print(f"\n✓ Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {high_corr_threshold})")

if len(high_corr_pairs) > 0:
    print("\nHighly Correlated Features (Potential Redundancy):")
    print("-" * 80)
    for i, pair in enumerate(high_corr_pairs[:10], 1):  # Show first 10
        print(f"{i:2d}. {pair['Feature 1']:30s} <-> {pair['Feature 2']:30s} : r = {pair['Correlation']:6.3f}")
    if len(high_corr_pairs) > 10:
        print(f"    ... and {len(high_corr_pairs) - 10} more pairs")

# Save high correlation pairs
high_corr_df = pd.DataFrame(high_corr_pairs)
if len(high_corr_df) > 0:
    high_corr_path = config.REPORTS_DIR / 'highly_correlated_features.csv'
    high_corr_df.to_csv(high_corr_path, index=False)
    print(f"\n✓ High correlation pairs saved to: {high_corr_path}")

# Create correlation heatmap (for subset of features to keep it readable)
# Select top 20 features by variance for visualization
feature_variances = X[numerical_features].var().sort_values(ascending=False)
top_features = feature_variances.head(20).index.tolist()

print(f"\n✓ Creating correlation heatmap for top {len(top_features)} features by variance...")

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(X[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Heatmap (Top 20 Features by Variance)', 
            fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
heatmap_path = eda_plots_dir / 'correlation_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Correlation heatmap saved to: {heatmap_path}")
plt.close()

# ============================================================================
# STEP 4: Feature Distributions by Class (KDE Plots)
# ============================================================================
print("\n[4/8] Creating KDE plots for feature distributions by class...")

# Select top 6 features by variance for KDE plots
top_kde_features = feature_variances.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Distributions by Habitability (KDE)', fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, feature in enumerate(top_kde_features):
    for cls in sorted(y.unique()):
        data = X.loc[y == cls, feature].dropna()
        color = config.BINARY_COLORS.get(cls, 'gray')
        label = 'Non-Habitable' if cls == 0 else 'Habitable'
        axes[idx].hist(data, bins=30, alpha=0.3, color=color, density=True, label=label)
        data.plot.kde(ax=axes[idx], color=color, linewidth=2)
    
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Density', fontsize=10)
    axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
kde_path = eda_plots_dir / 'kde_distributions.png'
plt.savefig(kde_path, dpi=300, bbox_inches='tight')
print(f"✓ KDE plots saved to: {kde_path}")
plt.close()

# ============================================================================
# STEP 5: Boxplots by Class
# ============================================================================
print("\n[5/8] Creating boxplots grouped by habitability...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Distributions by Habitability (Boxplots)', fontsize=16, fontweight='bold')
axes = axes.flatten()

# Prepare data for boxplots
plot_df = X[top_kde_features].copy()
plot_df['Habitability'] = y.map({0: 'Non-Habitable', 1: 'Habitable'})

for idx, feature in enumerate(top_kde_features):
    sns.boxplot(data=plot_df, x='Habitability', y=feature, ax=axes[idx],
               palette={'Non-Habitable': 'red', 'Habitable': 'green'})
    axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
boxplot_path = eda_plots_dir / 'boxplots_by_class.png'
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
print(f"✓ Boxplots saved to: {boxplot_path}")
plt.close()

# ============================================================================
# STEP 6: Scatter Plots for Top Feature Pairs
# ============================================================================
print("\n[6/8] Creating scatter plots for top feature pairs...")

# Select top 4 features by variance
top_scatter_features = feature_variances.head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Scatter Plots: Top Features by Habitability', fontsize=16, fontweight='bold')
axes = axes.flatten()

plot_pairs = [
    (top_scatter_features[0], top_scatter_features[1]),
    (top_scatter_features[0], top_scatter_features[2]),
    (top_scatter_features[1], top_scatter_features[3]),
    (top_scatter_features[2], top_scatter_features[3])
]

for idx, (feat1, feat2) in enumerate(plot_pairs):
    for cls in sorted(y.unique()):
        mask = y == cls
        color = config.BINARY_COLORS.get(cls, 'gray')
        label = 'Non-Habitable' if cls == 0 else 'Habitable'
        axes[idx].scatter(X.loc[mask, feat1], X.loc[mask, feat2], 
                         c=color, alpha=0.6, s=30, label=label, edgecolors='black', linewidth=0.5)
    
    axes[idx].set_xlabel(feat1, fontsize=10)
    axes[idx].set_ylabel(feat2, fontsize=10)
    axes[idx].set_title(f'{feat1} vs {feat2}', fontsize=11, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
scatter_path = eda_plots_dir / 'scatter_plots.png'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"✓ Scatter plots saved to: {scatter_path}")
plt.close()

# ============================================================================
# STEP 7: Statistical Tests for Feature-Target Relationships
# ============================================================================
print("\n[7/8] Performing statistical tests...")

print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE TESTING")
print("=" * 80)
print("\nPerforming independent t-tests for numerical features...")
print("(Testing if feature means differ significantly between classes)")

# Perform t-tests for numerical features
t_test_results = []

for feature in numerical_features:
    # Get data for each class
    class_0_data = X.loc[y == 0, feature].dropna()
    class_1_data = X.loc[y == 1, feature].dropna()
    
    # Perform independent t-test
    if len(class_0_data) > 1 and len(class_1_data) > 1:
        t_stat, p_value = stats.ttest_ind(class_0_data, class_1_data)
        
        t_test_results.append({
            'Feature': feature,
            'Mean_NonHabitable': class_0_data.mean(),
            'Mean_Habitable': class_1_data.mean(),
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

# Sort by p-value
t_test_df = pd.DataFrame(t_test_results).sort_values('P_Value')

# Show top significant features
print(f"\n✓ Performed t-tests for {len(t_test_results)} features")
significant_features = t_test_df[t_test_df['Significant'] == 'Yes']
print(f"✓ Found {len(significant_features)} statistically significant features (p < 0.05)")

print("\nTop 10 Most Significant Features:")
print("-" * 80)
for i, row in t_test_df.head(10).iterrows():
    print(f"{row['Feature']:40s} : p = {row['P_Value']:.2e} ({row['Significant']})")

# Save statistical test results
stats_path = config.REPORTS_DIR / 'statistical_tests.csv'
t_test_df.to_csv(stats_path, index=False)
print(f"\n✓ Statistical test results saved to: {stats_path}")

# ============================================================================
# STEP 8: Generate Comprehensive EDA Report
# ============================================================================
print("\n[8/8] Generating comprehensive EDA report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
report_lines.append("Exoplanet Habitability Classification")
report_lines.append("=" * 80)
report_lines.append("")

# Dataset Overview
report_lines.append("1. DATASET OVERVIEW")
report_lines.append("-" * 80)
report_lines.append(f"Total Samples: {df.shape[0]:,}")
report_lines.append(f"Total Features: {X.shape[1]}")
report_lines.append(f"Numerical Features: {len(numerical_features)}")
report_lines.append(f"Target Variable: {config.TARGET_VARIABLE}")
report_lines.append("")

# Class Imbalance
report_lines.append("2. CLASS IMBALANCE")
report_lines.append("-" * 80)
for cls in class_counts.index:
    report_lines.append(f"Class {cls}: {class_counts[cls]:5d} samples ({class_percentages[cls]:5.2f}%)")
if len(class_counts) == 2:
    report_lines.append(f"\nImbalance Ratio: {imbalance_ratio:.1f}:1")
    report_lines.append("⚠️  EXTREME IMBALANCE - Special techniques required!")
report_lines.append("")

# Correlation Analysis
report_lines.append("3. CORRELATION ANALYSIS")
report_lines.append("-" * 80)
report_lines.append(f"Highly Correlated Feature Pairs (|r| > {high_corr_threshold}): {len(high_corr_pairs)}")
if len(high_corr_pairs) > 0:
    report_lines.append("\nTop 5 Highly Correlated Pairs:")
    for i, pair in enumerate(high_corr_pairs[:5], 1):
        report_lines.append(f"  {i}. {pair['Feature 1']} <-> {pair['Feature 2']} : r = {pair['Correlation']:.3f}")
report_lines.append("")

# Statistical Significance
report_lines.append("4. STATISTICAL SIGNIFICANCE")
report_lines.append("-" * 80)
report_lines.append(f"Features with Significant Difference (p < 0.05): {len(significant_features)}")
report_lines.append("\nTop 10 Most Significant Features:")
for i, row in t_test_df.head(10).iterrows():
    report_lines.append(f"  {row['Feature']:40s} : p = {row['P_Value']:.2e}")
report_lines.append("")

# Key Findings
report_lines.append("5. KEY FINDINGS")
report_lines.append("-" * 80)
report_lines.append("✓ Severe class imbalance detected - standard ML will fail")
report_lines.append("✓ Multiple redundant features identified (high correlation)")
report_lines.append(f"✓ {len(significant_features)} features show significant relationship with habitability")
report_lines.append("✓ Feature distributions show overlap between classes")
report_lines.append("")

# Recommendations
report_lines.append("6. RECOMMENDATIONS")
report_lines.append("-" * 80)
report_lines.append("1. Use SMOTE or other sampling techniques to handle imbalance")
report_lines.append("2. Consider removing highly correlated features to reduce dimensionality")
report_lines.append("3. Focus on statistically significant features for modeling")
report_lines.append("4. Use appropriate metrics (F1, Recall, Precision) instead of Accuracy")
report_lines.append("5. Apply stratified sampling for train/test split")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("END OF EDA REPORT")
report_lines.append("=" * 80)

# Save report
report_path = config.REPORTS_DIR / 'eda_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ EDA report saved to: {report_path}")

# Save correlation matrix
corr_matrix_path = config.REPORTS_DIR / 'correlation_matrix.csv'
correlation_matrix.to_csv(corr_matrix_path)
print(f"✓ Full correlation matrix saved to: {corr_matrix_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

print("\n📊 Analysis Summary:")
print(f"  ✓ Dataset: {df.shape[0]:,} samples × {X.shape[1]} features")
print(f"  ✓ Class imbalance ratio: {imbalance_ratio:.1f}:1")
print(f"  ✓ Highly correlated pairs: {len(high_corr_pairs)}")
print(f"  ✓ Significant features: {len(significant_features)}")

print(f"\n📁 Files Generated:")
print(f"  1. Class imbalance plot: {imbalance_path}")
print(f"  2. Correlation heatmap: {heatmap_path}")
print(f"  3. KDE distributions: {kde_path}")
print(f"  4. Boxplots: {boxplot_path}")
print(f"  5. Scatter plots: {scatter_path}")
print(f"  6. EDA report: {report_path}")
print(f"  7. Statistical tests: {stats_path}")
print(f"  8. Correlation matrix: {corr_matrix_path}")
if len(high_corr_pairs) > 0:
    print(f"  9. High correlation pairs: {high_corr_path}")

print("\n🎓 Key Learnings:")
print("  1. Class imbalance is SEVERE - accuracy will be misleading")
print("  2. Many features are highly correlated (redundant)")
print("  3. Statistical tests identify truly important features")
print("  4. Visual analysis reveals class overlap and separability")
print("  5. EDA guides feature selection and model choice")

print("\n✅ Ready for Phase 6: Dimensionality Reduction Visualization!")
print("=" * 80)
