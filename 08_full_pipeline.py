"""
PHASE 9: FULL ML PIPELINE DEVELOPMENT
======================================

This script creates a complete, production-ready scikit-learn Pipeline that:
1. Loads the preprocessing pipeline (encoding + scaling)
2. Applies SMOTE sampling (only during training)
3. Trains the best performing model
4. Serializes the entire pipeline for deployment

CRITICAL: Pipeline ensures no data leakage and reproducible predictions!

LEARNING OBJECTIVES:
-------------------
- How to build end-to-end ML pipelines
- Proper integration of preprocessing and sampling
- Pipeline serialization for production
- Best practices for model deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix, average_precision_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
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
print("PHASE 9: FULL ML PIPELINE DEVELOPMENT")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data and Preprocessing Pipeline
# ============================================================================
print("\n[1/7] Loading data and preprocessing pipeline...")

features_path = config.PROCESSED_DATA_DIR / 'features_encoded_scaled.csv'
target_path = config.PROCESSED_DATA_DIR / 'target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)[config.TARGET_VARIABLE]

print(f"✓ Loaded features: {X.shape}")
print(f"✓ Loaded target: {y.shape}")

# Load preprocessing pipeline
preprocessing_pipeline_path = config.MODELS_DIR / 'preprocessing_pipeline.pkl'
preprocessing_pipeline = joblib.load(preprocessing_pipeline_path)
print(f"✓ Loaded preprocessing pipeline from: {preprocessing_pipeline_path}")

# Train/test split (same as before for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config.TEST_SIZE, 
    random_state=config.RANDOM_STATE,
    stratify=y
)

print(f"\n✓ Train/test split:")
print(f"  Training: {X_train.shape[0]:,} samples")
print(f"  Test: {X_test.shape[0]:,} samples")

# ============================================================================
# STEP 2: Analyze Best Model from Previous Phases
# ============================================================================
print("\n[2/7] Analyzing best model from SMOTE comparison...")

smote_results_path = config.REPORTS_DIR / 'smote_complete_results.csv'
results_df = pd.read_csv(smote_results_path)

# Filter out baseline
smote_results = results_df[results_df['Sampling'] != 'Baseline (No Sampling)']

# Find best model based on F1 score
best_idx = smote_results['F1'].idxmax()
best_result = smote_results.loc[best_idx]

print(f"\n🏆 Best Model Configuration:")
print(f"  Sampling Strategy: {best_result['Sampling']}")
print(f"  Model: {best_result['Model']}")
print(f"  F1 Score: {best_result['F1']:.4f}")
print(f"  Recall: {best_result['Recall']:.4f}")
print(f"  Precision: {best_result['Precision']:.4f}")
print(f"  ROC-AUC: {best_result['ROC-AUC']:.4f}")

# ============================================================================
# STEP 3: Define Pipeline Components
# ============================================================================
print("\n[3/7] Creating pipeline components...")

print("\n" + "=" * 80)
print("CONCEPT: What is a Pipeline?")
print("=" * 80)
print("A Pipeline chains multiple steps into a single object:")
print("  1. Ensures all steps are applied in correct order")
print("  2. Prevents data leakage (fit only on training data)")
print("  3. Simplifies code and reduces errors")
print("  4. Makes deployment easier (single object to save/load)")
print("  5. Enables proper cross-validation")
print("\n✅ Benefits:")
print("  ✓ Reproducible predictions")
print("  ✓ No manual step tracking")
print("  ✓ Automatic parameter tuning across all steps")
print("  ✓ Production-ready code")
print("=" * 80)

# Map sampling strategies
sampling_map = {
    'SMOTE': SMOTE(random_state=config.RANDOM_STATE),
    'Borderline-SMOTE': BorderlineSMOTE(random_state=config.RANDOM_STATE, kind='borderline-1'),
    'SMOTE+Tomek': SMOTETomek(random_state=config.RANDOM_STATE),
    'ADASYN': ADASYN(random_state=config.RANDOM_STATE)
}

# Map model configurations
model_map = {
    'Logistic Regression': LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS),
    'Linear SVM': SVC(**config.SVM_PARAMS, probability=True),
    'Random Forest': RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
}

# Get best components
best_sampler = sampling_map[best_result['Sampling']]
best_model = model_map[best_result['Model']]

print(f"\n✓ Selected Sampler: {best_result['Sampling']}")
print(f"✓ Selected Model: {best_result['Model']}")

# ============================================================================
# STEP 4: Build Complete Pipeline
# ============================================================================
print("\n[4/7] Building complete ML pipeline...")

# Create pipeline using imbalanced-learn's Pipeline (supports SMOTE)
full_pipeline = ImbPipeline([
    ('sampler', best_sampler),
    ('classifier', best_model)
])

print(f"\n✓ Pipeline created with {len(full_pipeline.steps)} steps:")
for i, (name, step) in enumerate(full_pipeline.steps, 1):
    print(f"  {i}. {name}: {type(step).__name__}")

# ============================================================================
# STEP 5: Train and Evaluate Pipeline
# ============================================================================
print("\n[5/7] Training and evaluating pipeline...")

print("\n⏳ Training pipeline on training data...")
full_pipeline.fit(X_train, y_train)
print("✓ Pipeline training complete!")

# Predictions
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
pr_auc = average_precision_score(y_test, y_pred_proba, average='weighted')

print(f"\n📊 Pipeline Performance on Test Set:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"  PR-AUC:    {pr_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📈 Confusion Matrix:")
print(cm)

# ============================================================================
# STEP 6: Cross-Validation
# ============================================================================
print("\n[6/7] Performing cross-validation...")

print("\n⏳ Running 5-fold stratified cross-validation...")
cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

# Cross-validation scores
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)

print(f"\n✓ Cross-Validation Results:")
print(f"  F1 Scores: {cv_scores}")
print(f"  Mean F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 7: Save Pipeline
# ============================================================================
print("\n[7/7] Saving complete pipeline...")

# Create pipeline directory
pipeline_dir = config.MODELS_DIR / 'pipeline'
pipeline_dir.mkdir(parents=True, exist_ok=True)

# Save pipeline
pipeline_path = pipeline_dir / 'full_pipeline.pkl'
joblib.dump(full_pipeline, pipeline_path)
print(f"✓ Pipeline saved to: {pipeline_path}")

# Save pipeline metadata
metadata = {
    'sampling_strategy': best_result['Sampling'],
    'model_type': best_result['Model'],
    'test_accuracy': accuracy,
    'test_f1': f1,
    'test_recall': recall,
    'test_precision': precision,
    'test_roc_auc': roc_auc,
    'cv_mean_f1': cv_scores.mean(),
    'cv_std_f1': cv_scores.std(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'n_features': X_train.shape[1],
    'random_state': config.RANDOM_STATE
}

metadata_df = pd.DataFrame([metadata])
metadata_path = pipeline_dir / 'pipeline_metadata.csv'
metadata_df.to_csv(metadata_path, index=False)
print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# STEP 8: Test Pipeline Loading and Prediction
# ============================================================================
print("\n[8/8] Testing pipeline loading and prediction...")

# Load pipeline
loaded_pipeline = joblib.load(pipeline_path)
print("✓ Pipeline loaded successfully!")

# Test prediction on a sample
sample_idx = 0
X_sample = X_test.iloc[[sample_idx]]
y_sample = y_test.iloc[sample_idx]

prediction = loaded_pipeline.predict(X_sample)[0]
prediction_proba = loaded_pipeline.predict_proba(X_sample)[0]

print(f"\n🧪 Test Prediction:")
print(f"  True Label: {y_sample}")
print(f"  Predicted:  {prediction}")
print(f"  Probabilities: {prediction_proba}")
print(f"  Match: {'✓ Correct' if prediction == y_sample else '✗ Incorrect'}")

# ============================================================================
# VISUALIZATION: Pipeline Architecture
# ============================================================================
print("\n[Bonus] Creating pipeline architecture visualization...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Pipeline steps
steps = [
    "Input Data\n(Exoplanet Features)",
    f"SMOTE Sampling\n({best_result['Sampling']})",
    f"Classification\n({best_result['Model']})",
    "Prediction Output\n(Habitability Class)"
]

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
y_positions = [0.8, 0.6, 0.4, 0.2]

for i, (step, color, y_pos) in enumerate(zip(steps, colors, y_positions)):
    # Draw box
    bbox = dict(boxstyle='round,pad=0.8', facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.text(0.5, y_pos, step, ha='center', va='center', fontsize=14, 
            fontweight='bold', bbox=bbox, color='white')
    
    # Draw arrow (except for last step)
    if i < len(steps) - 1:
        ax.annotate('', xy=(0.5, y_positions[i+1] + 0.05), xytext=(0.5, y_pos - 0.05),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Add metrics
metrics_text = f"Performance Metrics:\n"
metrics_text += f"F1 Score: {f1:.4f}\n"
metrics_text += f"Recall: {recall:.4f}\n"
metrics_text += f"ROC-AUC: {roc_auc:.4f}"

ax.text(0.98, 0.5, metrics_text, ha='right', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
        family='monospace')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Complete ML Pipeline Architecture', fontsize=18, fontweight='bold', pad=20)

pipeline_viz_path = config.PLOTS_DIR / 'pipeline_architecture.png'
plt.savefig(pipeline_viz_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Pipeline visualization saved to: {pipeline_viz_path}")
plt.close()

# ============================================================================
# GENERATE REPORT
# ============================================================================
print("\n[Report] Generating pipeline report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("FULL ML PIPELINE - COMPREHENSIVE REPORT")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("1. PIPELINE CONFIGURATION")
report_lines.append("-" * 80)
report_lines.append(f"Sampling Strategy: {best_result['Sampling']}")
report_lines.append(f"Model Type: {best_result['Model']}")
report_lines.append(f"Number of Features: {X_train.shape[1]}")
report_lines.append(f"Training Samples: {len(X_train):,}")
report_lines.append(f"Test Samples: {len(X_test):,}")
report_lines.append("")

report_lines.append("2. PIPELINE STEPS")
report_lines.append("-" * 80)
for i, (name, step) in enumerate(full_pipeline.steps, 1):
    report_lines.append(f"{i}. {name}: {type(step).__name__}")
report_lines.append("")

report_lines.append("3. PERFORMANCE METRICS")
report_lines.append("-" * 80)
report_lines.append(f"Test Set Performance:")
report_lines.append(f"  Accuracy:  {accuracy:.4f}")
report_lines.append(f"  Precision: {precision:.4f}")
report_lines.append(f"  Recall:    {recall:.4f}")
report_lines.append(f"  F1 Score:  {f1:.4f}")
report_lines.append(f"  ROC-AUC:   {roc_auc:.4f}")
report_lines.append(f"  PR-AUC:    {pr_auc:.4f}")
report_lines.append("")
report_lines.append(f"Cross-Validation (5-Fold):")
report_lines.append(f"  Mean F1:   {cv_scores.mean():.4f}")
report_lines.append(f"  Std F1:    {cv_scores.std():.4f}")
report_lines.append(f"  95% CI:    [{cv_scores.mean() - 2*cv_scores.std():.4f}, {cv_scores.mean() + 2*cv_scores.std():.4f}]")
report_lines.append("")

report_lines.append("4. CONFUSION MATRIX")
report_lines.append("-" * 80)
report_lines.append(f"True Negatives:  {cm[0, 0]}")
report_lines.append(f"False Positives: {cm[0, 1]}")
report_lines.append(f"False Negatives: {cm[1, 0]}")
report_lines.append(f"True Positives:  {cm[1, 1]}")
report_lines.append("")

report_lines.append("5. DEPLOYMENT ARTIFACTS")
report_lines.append("-" * 80)
report_lines.append(f"Pipeline File: {pipeline_path}")
report_lines.append(f"Metadata File: {metadata_path}")
report_lines.append(f"File Size: {pipeline_path.stat().st_size / 1024:.2f} KB")
report_lines.append("")

report_lines.append("6. USAGE INSTRUCTIONS")
report_lines.append("-" * 80)
report_lines.append("To use this pipeline in production:")
report_lines.append("  1. Load pipeline: pipeline = joblib.load('full_pipeline.pkl')")
report_lines.append("  2. Prepare input: X_new = pd.DataFrame([features])")
report_lines.append("  3. Predict: prediction = pipeline.predict(X_new)")
report_lines.append("  4. Get probabilities: proba = pipeline.predict_proba(X_new)")
report_lines.append("")

report_lines.append("7. KEY FEATURES")
report_lines.append("-" * 80)
report_lines.append("✓ End-to-end preprocessing and prediction")
report_lines.append("✓ Automatic SMOTE sampling during training")
report_lines.append("✓ No data leakage (sampling only on training data)")
report_lines.append("✓ Reproducible predictions")
report_lines.append("✓ Single object for deployment")
report_lines.append("✓ Cross-validated performance")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("END OF PIPELINE REPORT")
report_lines.append("=" * 80)

# Save report
report_path = config.REPORTS_DIR / 'full_pipeline_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Pipeline report saved to: {report_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 9: FULL PIPELINE DEVELOPMENT COMPLETE")
print("=" * 80)

print(f"\n🏆 Best Configuration:")
print(f"  Sampling: {best_result['Sampling']}")
print(f"  Model: {best_result['Model']}")

print(f"\n📊 Performance:")
print(f"  Test F1: {f1:.4f}")
print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n📁 Files Generated:")
print(f"  1. Pipeline: {pipeline_path}")
print(f"  2. Metadata: {metadata_path}")
print(f"  3. Visualization: {pipeline_viz_path}")
print(f"  4. Report: {report_path}")

print("\n🎓 Key Learnings:")
print("  1. Pipelines chain multiple steps into single object")
print("  2. Prevents data leakage automatically")
print("  3. Simplifies deployment (one file to save/load)")
print("  4. Enables proper cross-validation")
print("  5. Makes code more maintainable and reproducible")

print("\n✅ Phase 9 Complete! Ready for Phase 10 (Model Evaluation)")
print("=" * 80)
