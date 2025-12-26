import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load Data & Re-create Test Set
df = pd.read_csv('phl_exoplanet_catalog.csv')

# ensuring we use the exact columns the model was trained on
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]
target_col = 'P_HABITABLE'

# Clean
df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(df_clean[feature_cols]), columns=feature_cols)
y = df_clean[target_col]

# Split (Same random_state=42 ensures exact test set match)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Load the Champion Model (Pipeline)
with open('final_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

print("Model Pipeline Loaded. Steps:", model_pipeline.named_steps.keys())

# Get Predictions
# The pipeline automatically handles Imputation -> Prediction
# (SMOTE is correctly skipped during prediction)
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test) # Shape: (n_samples, 3)

# Calculate "Habitability Score" (Prob of Class 1 + Class 2)
habitability_scores = y_prob[:, 1] + y_prob[:, 2]

# --- PLOTTING SETUP (2x2 Grid) ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.3)

# --- PLOT 1: CONFUSION MATRIX ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0])
axes[0, 0].set_title("1. Final Confusion Matrix (SMOTE + XGBoost)")
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("Actual Label")
axes[0, 0].set_xticklabels(['0: Non-Hab', '1: Conservative', '2: Optimistic'])
axes[0, 0].set_yticklabels(['0: Non-Hab', '1: Conservative', '2: Optimistic'])

# --- PLOT 2: FEATURE IMPORTANCE ---
# CRITICAL CHANGE: We must extract the 'model' step from the pipeline
xgb_classifier = model_pipeline.named_steps['model']
importances = xgb_classifier.feature_importances_
indices = np.argsort(importances)
features = X.columns

axes[0, 1].barh(range(len(indices)), importances[indices], align='center', color='#2ca02c')
axes[0, 1].set_yticks(range(len(indices)))
axes[0, 1].set_yticklabels([features[i] for i in indices])
axes[0, 1].set_title("2. Feature Importance (What drives Habitability?)")
axes[0, 1].set_xlabel("Relative Importance")

# --- PLOT 3: HABITABILITY SCORE DISTRIBUTION ---
sns.histplot(
    x=habitability_scores, hue=y_test.astype(str), 
    element="step", stat="density", common_norm=False, 
    palette={'0': 'red', '1': 'green', '2': 'blue'}, ax=axes[1, 0]
)
axes[1, 0].set_title("3. Habitability Score Distribution")
axes[1, 0].set_xlabel("Predicted Habitability Score (0.0 to 1.0)")
axes[1, 0].set_ylabel("Density")
axes[1, 0].legend(title='Actual Class', labels=['2: Optimistic', '1: Conservative', '0: Non-Hab'])

# --- PLOT 4: ROC CURVES (Multiclass) ---
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]
colors = ['red', 'green', 'blue']
class_names = ['Non-Habitable', 'Conservative', 'Optimistic']

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=2)
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('4. ROC Curve (Performance per Class)')
axes[1, 1].legend(loc="lower right")

plt.show()
print("✅ Plots Generated.")