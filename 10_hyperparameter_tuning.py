"""
PHASE 11: HYPERPARAMETER TUNING
================================

This script performs hyperparameter tuning using GridSearchCV
to optimize the best model's performance.

LEARNING OBJECTIVES:
-------------------
- Understanding hyperparameter tuning
- Using GridSearchCV with pipelines
- Avoiding overfitting during tuning
- Selecting optimal parameters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

import config

print("=" * 80)
print("PHASE 11: HYPERPARAMETER TUNING")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")
features_path = config.PROCESSED_DATA_DIR / 'features_encoded_scaled.csv'
target_path = config.PROCESSED_DATA_DIR / 'target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)[config.TARGET_VARIABLE]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)

print(f"✓ Training samples: {len(X_train):,}")
print(f"✓ Test samples: {len(X_test):,}")

# Create pipeline
print("\n[2/4] Setting up hyperparameter grid...")

pipeline = ImbPipeline([
    ('sampler', SMOTE(random_state=config.RANDOM_STATE)),
    ('classifier', SVC(random_state=config.RANDOM_STATE, probability=True))
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__class_weight': ['balanced', None]
}

print(f"✓ Parameter grid defined with {np.prod([len(v) for v in param_grid.values()])} combinations")

# Grid search
print("\n[3/4] Performing grid search (this may take a while)...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Grid search complete!")
print(f"\n🏆 Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n📊 Best CV F1 Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
print("\n[4/4] Evaluating tuned model on test set...")
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

y_pred = grid_search.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average='weighted')
test_acc = accuracy_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred, average='weighted')
test_precision = precision_score(y_test, y_pred, average='weighted')

print(f"\n📊 Test Set Performance:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

# Save tuned model
tuned_model_path = config.MODELS_DIR / 'pipeline' / 'tuned_pipeline.pkl'
joblib.dump(grid_search.best_estimator_, tuned_model_path)
print(f"\n✓ Tuned model saved to: {tuned_model_path}")

# Save results
results_df = pd.DataFrame(grid_search.cv_results_)
results_path = config.REPORTS_DIR / 'hyperparameter_tuning_results.csv'
results_df.to_csv(results_path, index=False)
print(f"✓ Tuning results saved to: {results_path}")

print("\n" + "=" * 80)
print("PHASE 11 COMPLETE: Hyperparameter Tuning")
print("=" * 80)
print(f"✅ Tuned model achieves {test_f1:.2%} F1 score")
print("=" * 80)
