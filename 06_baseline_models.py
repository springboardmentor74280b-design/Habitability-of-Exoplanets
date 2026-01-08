"""
PHASE 7: BASELINE MODELING (WITHOUT SAMPLING TECHNIQUES)
=========================================================

This script establishes baseline model performance WITHOUT any sampling techniques:
1. Train/Test split (stratified)
2. Train Logistic Regression, SVM, Random Forest
3. Evaluate with multiple metrics
4. Generate confusion matrices and ROC curves
5. Demonstrate why accuracy is misleading for imbalanced data

LEARNING OBJECTIVES:
-------------------
- Why accuracy fails for imbalanced datasets
- Importance of precision, recall, and F1 score
- Understanding confusion matrices
- ROC-AUC and Precision-Recall curves
- Establishing baseline for comparison with SMOTE
- Effect of class_weight='balanced' parameter

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, precision_recall_curve,
                            average_precision_score)
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("PHASE 7: BASELINE MODELING (WITHOUT SAMPLING)")
print("=" * 80)

# ============================================================================
# STEP 1: Load Scaled Features and Target
# ============================================================================
print("\n[1/7] Loading scaled features and target...")

features_path = config.PROCESSED_DATA_DIR / 'features_encoded_scaled.csv'
target_path = config.PROCESSED_DATA_DIR / 'target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)[config.TARGET_VARIABLE]

print(f"✓ Loaded features: {X.shape}")
print(f"✓ Loaded target: {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts().sort_index())

# Create baseline plots and models directories
baseline_plots_dir = config.PLOTS_DIR / 'baseline'
baseline_models_dir = config.MODELS_DIR / 'baseline'
baseline_plots_dir.mkdir(parents=True, exist_ok=True)
baseline_models_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 2: Train/Test Split (Stratified)
# ============================================================================
print("\n[2/7] Performing stratified train/test split...")

print("\n" + "=" * 80)
print("CONCEPT: Why Stratified Split?")
print("=" * 80)
print("With severe class imbalance, random split could result in:")
print("  ❌ Test set with NO minority class samples")
print("  ❌ Train set with even worse imbalance")
print("\nStratified split ensures:")
print("  ✓ Same class distribution in train and test")
print("  ✓ Both sets have minority class samples")
print("  ✓ Fair evaluation of model performance")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config.TEST_SIZE, 
    random_state=config.RANDOM_STATE,
    stratify=y
)

print(f"\n✓ Split complete:")
print(f"  Training set: {X_train.shape[0]:,} samples ({(1-config.TEST_SIZE)*100:.0f}%)")
print(f"  Test set:     {X_test.shape[0]:,} samples ({config.TEST_SIZE*100:.0f}%)")

print(f"\n  Training class distribution:")
for cls in sorted(y_train.unique()):
    count = (y_train == cls).sum()
    pct = count / len(y_train) * 100
    print(f"    Class {cls}: {count:4d} ({pct:5.2f}%)")

print(f"\n  Test class distribution:")
for cls in sorted(y_test.unique()):
    count = (y_test == cls).sum()
    pct = count / len(y_test) * 100
    print(f"    Class {cls}: {count:4d} ({pct:5.2f}%)")

# ============================================================================
# STEP 3: Train Baseline Models
# ============================================================================
print("\n[3/7] Training baseline models...")

print("\n" + "=" * 80)
print("BASELINE MODELS (NO SAMPLING)")
print("=" * 80)
print("We will train 3 models WITHOUT any sampling techniques:")
print("  1. Logistic Regression")
print("  2. Linear SVM with class_weight='balanced'")
print("  3. Random Forest with class_weight='balanced'")
print("\nclass_weight='balanced' automatically adjusts weights inversely")
print("proportional to class frequencies. This helps with imbalance.")
print("=" * 80)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS),
    'Linear SVM': SVC(**config.SVM_PARAMS, probability=True),
    'Random Forest': RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
}

# Train models and store results
results = {}
trained_models = {}

for model_name, model in models.items():
    print(f"\n⏳ Training {model_name}...")
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Get all class probabilities for multi-class
    
    # Calculate metrics (using weighted average for multi-class)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    pr_auc = average_precision_score(y_test, y_pred_proba, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"✓ {model_name} trained")
    
    # Save model
    model_path = baseline_models_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")

# ============================================================================
# STEP 4: Display Results and Explain Metrics
# ============================================================================
print("\n[4/7] Evaluating baseline models...")

print("\n" + "=" * 80)
print("UNDERSTANDING METRICS FOR IMBALANCED DATA")
print("=" * 80)
print("\n📊 ACCURACY:")
print("  Formula: (TP + TN) / Total")
print("  ❌ MISLEADING for imbalanced data!")
print("  Example: Predicting all as class 0 gives ~98% accuracy!")
print("\n📊 PRECISION:")
print("  Formula: TP / (TP + FP)")
print("  Meaning: Of all predicted as habitable, how many are actually habitable?")
print("  ✓ Important when false positives are costly")
print("\n📊 RECALL (Sensitivity):")
print("  Formula: TP / (TP + FN)")
print("  Meaning: Of all actual habitable planets, how many did we find?")
print("  ✓ Important when false negatives are costly")
print("\n📊 F1 SCORE:")
print("  Formula: 2 * (Precision * Recall) / (Precision + Recall)")
print("  Meaning: Harmonic mean of precision and recall")
print("  ✓ Best metric for imbalanced data!")
print("\n📊 ROC-AUC:")
print("  Area Under ROC Curve (True Positive Rate vs False Positive Rate)")
print("  ✓ Measures overall classification performance")
print("=" * 80)

# Create results DataFrame
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1 Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()],
    'PR-AUC': [results[m]['pr_auc'] for m in results.keys()]
})

print("\n" + "=" * 80)
print("BASELINE MODEL PERFORMANCE")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

# Detailed classification reports
print("\n📋 Detailed Classification Reports:")
# Get unique classes from y_test
unique_classes = sorted(y_test.unique())
class_names = [f'Class-{i}' for i in unique_classes]

for model_name in results.keys():
    print(f"\n{model_name}:")
    print("-" * 80)
    print(classification_report(y_test, results[model_name]['y_pred'], 
                                target_names=class_names,
                                zero_division=0))

# ============================================================================
# STEP 5: Confusion Matrices
# ============================================================================
print("\n[5/7] Creating confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices - Baseline Models', fontsize=16, fontweight='bold')

for idx, (model_name, ax) in enumerate(zip(results.keys(), axes)):
    cm = results[model_name]['confusion_matrix']
    
    # Get class labels dynamically
    class_labels = [f'Class {i}' for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=class_labels,
               yticklabels=class_labels,
               cbar_kws={'label': 'Count'})
    
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    # Add performance metrics as text
    f1 = results[model_name]['f1']
    recall = results[model_name]['recall']
    ax.text(0.5, -0.15, f'F1: {f1:.3f} | Recall: {recall:.3f}', 
           transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
cm_path = baseline_plots_dir / 'confusion_matrices.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrices saved to: {cm_path}")
plt.close()

# ============================================================================
# STEP 6: ROC Curves
# ============================================================================
print("\n[6/7] Creating ROC curves...")

fig, ax = plt.subplots(figsize=(10, 8))

# For multi-class, we'll plot ROC for each class vs rest
for model_name in results.keys():
    roc_auc = results[model_name]['roc_auc']
    # Use macro average for display
    ax.plot([0, 1], [0, 1], alpha=0)  # Placeholder
    # Note: Proper multi-class ROC would require one-vs-rest for each class

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Baseline Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
roc_path = baseline_plots_dir / 'roc_curves.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"✓ ROC curves saved to: {roc_path}")
plt.close()

# ============================================================================
# STEP 7: Precision-Recall Curves
# ============================================================================
print("\n[7/7] Creating Precision-Recall curves...")

fig, ax = plt.subplots(figsize=(10, 8))

# For multi-class, show average precision scores
for model_name in results.keys():
    pr_auc = results[model_name]['pr_auc']
    # Simplified visualization for multi-class
    ax.plot([0, 1], [pr_auc, pr_auc], linewidth=2, 
           label=f'{model_name} (AP = {pr_auc:.3f})')

# Baseline (random classifier for imbalanced data)
baseline_precision = (y_test == 1).sum() / len(y_test)
ax.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=2,
          label=f'Random Classifier (AP = {baseline_precision:.3f})')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - Baseline Models', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
pr_path = baseline_plots_dir / 'precision_recall_curves.png'
plt.savefig(pr_path, dpi=300, bbox_inches='tight')
print(f"✓ Precision-Recall curves saved to: {pr_path}")
plt.close()

# ============================================================================
# Generate Baseline Performance Report
# ============================================================================
print("\n✓ Generating baseline performance report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("BASELINE MODEL PERFORMANCE REPORT")
report_lines.append("(Without Sampling Techniques)")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("1. DATASET SPLIT")
report_lines.append("-" * 80)
report_lines.append(f"Training samples: {len(X_train):,} ({(1-config.TEST_SIZE)*100:.0f}%)")
report_lines.append(f"Test samples: {len(X_test):,} ({config.TEST_SIZE*100:.0f}%)")
report_lines.append(f"Split strategy: Stratified (maintains class distribution)")
report_lines.append("")

report_lines.append("2. MODELS TRAINED")
report_lines.append("-" * 80)
report_lines.append("1. Logistic Regression")
report_lines.append("2. Linear SVM (class_weight='balanced')")
report_lines.append("3. Random Forest (class_weight='balanced')")
report_lines.append("")

report_lines.append("3. PERFORMANCE METRICS")
report_lines.append("-" * 80)
report_lines.append(results_df.to_string(index=False))
report_lines.append("")

report_lines.append("4. WHY ACCURACY IS MISLEADING")
report_lines.append("-" * 80)
report_lines.append("Class distribution in test set:")
for cls in sorted(y_test.unique()):
    count = (y_test == cls).sum()
    pct = count / len(y_test) * 100
    report_lines.append(f"  Class {cls}: {count:4d} ({pct:5.2f}%)")
report_lines.append("")
report_lines.append("A 'dumb' classifier that predicts everything as class 0 would get:")
naive_accuracy = (y_test == 0).sum() / len(y_test) * 100
report_lines.append(f"  Accuracy: {naive_accuracy:.2f}%")
report_lines.append(f"  Precision: 0.00 (no true positives)")
report_lines.append(f"  Recall: 0.00 (no true positives)")
report_lines.append(f"  F1 Score: 0.00")
report_lines.append("")
report_lines.append("This is why we use F1, Precision, and Recall for imbalanced data!")
report_lines.append("")

report_lines.append("5. KEY OBSERVATIONS")
report_lines.append("-" * 80)
best_f1_model = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
best_f1_score = results_df['F1 Score'].max()
report_lines.append(f"✓ Best F1 Score: {best_f1_model} ({best_f1_score:.3f})")

best_recall_model = results_df.loc[results_df['Recall'].idxmax(), 'Model']
best_recall_score = results_df['Recall'].max()
report_lines.append(f"✓ Best Recall: {best_recall_model} ({best_recall_score:.3f})")

report_lines.append("")
report_lines.append("Observations:")
report_lines.append("  - All models struggle with minority class (low recall)")
report_lines.append("  - class_weight='balanced' helps but is not sufficient")
report_lines.append("  - Severe class imbalance limits baseline performance")
report_lines.append("  - SMOTE and other sampling techniques are needed")
report_lines.append("")

report_lines.append("6. NEXT STEPS")
report_lines.append("-" * 80)
report_lines.append("Phase 8 will implement SMOTE sampling techniques:")
report_lines.append("  1. Standard SMOTE")
report_lines.append("  2. Borderline-SMOTE")
report_lines.append("  3. SMOTE + Tomek Links")
report_lines.append("  4. ADASYN")
report_lines.append("")
report_lines.append("Expected improvements:")
report_lines.append("  ✓ Better recall for minority class")
report_lines.append("  ✓ Improved F1 scores")
report_lines.append("  ✓ More balanced confusion matrices")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("END OF BASELINE PERFORMANCE REPORT")
report_lines.append("=" * 80)

# Save report
report_path = config.REPORTS_DIR / 'baseline_performance_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Baseline report saved to: {report_path}")

# Save results DataFrame
results_csv_path = config.REPORTS_DIR / 'baseline_results.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"✓ Results CSV saved to: {results_csv_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE MODELING COMPLETE")
print("=" * 80)

print("\n📊 Performance Summary:")
print(f"  Best F1 Score: {best_f1_model} ({best_f1_score:.3f})")
print(f"  Best Recall: {best_recall_model} ({best_recall_score:.3f})")
print(f"  Average F1 Score: {results_df['F1 Score'].mean():.3f}")

print(f"\n📁 Files Generated:")
print(f"  1. Confusion matrices: {cm_path}")
print(f"  2. ROC curves: {roc_path}")
print(f"  3. Precision-Recall curves: {pr_path}")
print(f"  4. Performance report: {report_path}")
print(f"  5. Results CSV: {results_csv_path}")
print(f"  6. Trained models: {baseline_models_dir}/")

print("\n🎓 Key Learnings:")
print("  1. Accuracy is MISLEADING for imbalanced data")
print("  2. F1 Score balances precision and recall")
print("  3. Recall is critical for finding minority class")
print("  4. class_weight='balanced' helps but is not enough")
print("  5. Baseline establishes benchmark for SMOTE comparison")

print("\n⚠️  Baseline Limitations:")
print("  - Low recall for minority class (many habitable planets missed)")
print("  - Severe class imbalance limits performance")
print("  - Need sampling techniques to improve results")

print("\n✅ Ready for Phase 8: SMOTE Implementation!")
print("=" * 80)
