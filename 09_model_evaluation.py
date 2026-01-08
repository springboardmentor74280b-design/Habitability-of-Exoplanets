"""
PHASE 10: COMPREHENSIVE MODEL EVALUATION
=========================================

This script performs comprehensive evaluation of the best model:
1. Detailed performance metrics
2. ROC and PR curves
3. Confusion matrix visualization
4. Classification report
5. Error analysis

LEARNING OBJECTIVES:
-------------------
- Understanding evaluation metrics for imbalanced data
- Visualizing model performance
- Identifying model strengths and weaknesses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, auc,
                            precision_recall_curve, average_precision_score,
                            confusion_matrix, classification_report)
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import config

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("PHASE 10: COMPREHENSIVE MODEL EVALUATION")
print("=" * 80)

# Load data
print("\n[1/5] Loading data and pipeline...")
features_path = config.PROCESSED_DATA_DIR / 'features_encoded_scaled.csv'
target_path = config.PROCESSED_DATA_DIR / 'target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)[config.TARGET_VARIABLE]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)

# Load pipeline
pipeline_path = config.MODELS_DIR / 'pipeline' / 'full_pipeline.pkl'
pipeline = joblib.load(pipeline_path)
print(f"✓ Loaded pipeline from: {pipeline_path}")

# Make predictions
print("\n[2/5] Generating predictions...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

print(f"\n📊 Overall Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# Classification report
print("\n[3/5] Generating classification report...")
print("\n" + classification_report(y_test, y_pred, target_names=['Non-Habitable', 'Habitable', 'Optimistic']))

# Create visualizations
print("\n[4/5] Creating evaluation visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, :2])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_xticklabels(['Non-Hab', 'Hab', 'Opt-Hab'])
ax1.set_yticklabels(['Non-Hab', 'Hab', 'Opt-Hab'])

# 2. ROC Curves (One-vs-Rest)
ax2 = fig.add_subplot(gs[1, 0])
n_classes = len(np.unique(y))
for i in range(n_classes):
    y_test_binary = (y_test == i).astype(int)
    y_score = y_pred_proba[:, i]
    fpr, tpr, _ = roc_curve(y_test_binary, y_score)
    roc_auc_class = auc(fpr, tpr)
    ax2.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc_class:.3f})')

ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curves (One-vs-Rest)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Precision-Recall Curves
ax3 = fig.add_subplot(gs[1, 1])
for i in range(n_classes):
    y_test_binary = (y_test == i).astype(int)
    y_score = y_pred_proba[:, i]
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_score)
    pr_auc = average_precision_score(y_test_binary, y_score)
    ax3.plot(recall_curve, precision_curve, lw=2, label=f'Class {i} (AP = {pr_auc:.3f})')

ax3.set_xlabel('Recall', fontsize=11)
ax3.set_ylabel('Precision', fontsize=11)
ax3.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Class Distribution
ax4 = fig.add_subplot(gs[1, 2])
class_counts = pd.Series(y_test).value_counts().sort_index()
colors = ['#e74c3c', '#2ecc71', '#3498db']
ax4.bar(class_counts.index, class_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Class', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(['Non-Hab', 'Hab', 'Opt-Hab'])
ax4.grid(True, alpha=0.3, axis='y')

# 5. Per-Class Metrics
ax5 = fig.add_subplot(gs[2, :])
per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

x = np.arange(n_classes)
width = 0.25
ax5.bar(x - width, per_class_precision, width, label='Precision', color='#3498db', alpha=0.8)
ax5.bar(x, per_class_recall, width, label='Recall', color='#2ecc71', alpha=0.8)
ax5.bar(x + width, per_class_f1, width, label='F1 Score', color='#e74c3c', alpha=0.8)

ax5.set_xlabel('Class', fontsize=11, fontweight='bold')
ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
ax5.set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(['Non-Habitable', 'Habitable', 'Optimistic'])
ax5.legend(loc='lower right', fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 1.1])

plt.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold', y=0.995)

eval_viz_path = config.PLOTS_DIR / 'model_evaluation_comprehensive.png'
plt.savefig(eval_viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Evaluation visualization saved to: {eval_viz_path}")
plt.close()

# Save evaluation report
print("\n[5/5] Generating evaluation report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("1. OVERALL PERFORMANCE")
report_lines.append("-" * 80)
report_lines.append(f"Accuracy:  {accuracy:.4f}")
report_lines.append(f"Precision: {precision:.4f}")
report_lines.append(f"Recall:    {recall:.4f}")
report_lines.append(f"F1 Score:  {f1:.4f}")
report_lines.append(f"ROC-AUC:   {roc_auc:.4f}")
report_lines.append("")
report_lines.append("2. CONFUSION MATRIX")
report_lines.append("-" * 80)
report_lines.append(str(cm))
report_lines.append("")
report_lines.append("3. CLASSIFICATION REPORT")
report_lines.append("-" * 80)
report_lines.append(classification_report(y_test, y_pred, target_names=['Non-Habitable', 'Habitable', 'Optimistic']))
report_lines.append("")
report_lines.append("=" * 80)

report_path = config.REPORTS_DIR / 'model_evaluation_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Evaluation report saved to: {report_path}")

print("\n" + "=" * 80)
print("PHASE 10 COMPLETE: Model Evaluation")
print("=" * 80)
print(f"\n✅ Model achieves {f1:.2%} F1 score on test set")
print(f"📁 Visualization: {eval_viz_path}")
print(f"📁 Report: {report_path}")
print("=" * 80)
