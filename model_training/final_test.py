import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, 
    f1_score, classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import LabelBinarizer

# --- 1. SETUP & LOADING ---
print("📊 Starting Scientific Evaluation...")

# Load Ground Truth (PHL)
df_phl = pd.read_csv('phl_exoplanet_catalog.csv')
# Keep only Name and Label
df_truth = df_phl[['P_NAME', 'P_HABITABLE']].copy()
# Normalize Names for Matching (Lowercase, strip spaces)
df_truth['P_NAME_JOIN'] = df_truth['P_NAME'].str.strip().str.lower()

# Load Test Data (NASA Upload)
filename = 'PSCompPars_2025.12.24_23.44.10.csv'
df_nasa = pd.read_csv(filename, comment='#')
df_nasa.columns = df_nasa.columns.str.strip().str.lower()

# Rename NASA columns to match Model
col_map = {
    'pl_name': 'P_NAME', 'pl_bmasse': 'P_MASS_EST', 'pl_rade': 'P_RADIUS_EST',
    'pl_orbper': 'P_PERIOD', 'pl_orbsmax': 'P_DISTANCE', 'pl_eqt': 'P_TEMP_EQUIL',
    'st_teff': 'S_TEMPERATURE', 'st_lum': 'S_LUMINOSITY', 'st_rad': 'S_RADIUS',
    'st_mass': 'S_MASS', 'st_met': 'S_METALLICITY'
}
df_nasa.rename(columns=col_map, inplace=True)
df_nasa['P_NAME_JOIN'] = df_nasa['P_NAME'].str.strip().str.lower()

# --- 2. THE MERGE (CROSS-DATASET MATCHING) ---
print("🔗 Merging PHL Labels with NASA Physics...")
# We only test planets that exist in BOTH catalogs
df_test = pd.merge(df_nasa, df_truth, on='P_NAME_JOIN', how='inner', suffixes=('', '_truth'))

# Data Cleaning (Same as before)
if 'S_LUMINOSITY' in df_test.columns and df_test['S_LUMINOSITY'].max() < 100:
    df_test['S_LUMINOSITY'] = 10**df_test['S_LUMINOSITY']

feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]

# Ensure features exist
for col in feature_cols:
    if col not in df_test.columns:
        df_test[col] = np.nan

# Define X (Features from NASA) and y (Labels from PHL)
X_test = df_test[feature_cols]
y_true = df_test['P_HABITABLE']

print(f"✅ Successfully matched {len(df_test)} planets for validation.")

# --- 3. RUN MODEL ---
print("🧠 Loading Model & Predicting...")
with open('final_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test) # Needed for ROC-AUC

# --- 4. CALCULATE METRICS ---
print("\n" + "="*40)
print("     SCIENTIFIC PERFORMANCE REPORT")
print("="*40)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc:.2%}")

# Macro Average (Treats all classes equally - important for rare Habitable planets)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f"Precision (Macro): {prec:.2f}")
print(f"Recall (Macro):    {rec:.2f}")
print(f"F1-Score (Macro):  {f1:.2f}")

# Detailed Report
print("\n--- Detailed Classification Report ---")
print(classification_report(y_true, y_pred, target_names=['Non-Habitable', 'Conservative', 'Optimistic']))

# --- 5. VISUALIZATIONS ---

# A. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Hab', 'Conservative', 'Optimistic'],
            yticklabels=['Non-Hab', 'Conservative', 'Optimistic'])
plt.title('Confusion Matrix: NASA Data vs PHL Labels')
plt.xlabel('AI Prediction (Based on NASA Data)')
plt.ylabel('True Label (PHL Catalog)')
plt.tight_layout()
plt.savefig('nasa_confusion_matrix.png') # Save for mentor
plt.show()

# B. ROC-AUC Curve (Multi-class One-vs-Rest)
# We binarize the labels for multiclass ROC
lb = LabelBinarizer()
lb.fit(y_true)
y_true_bin = lb.transform(y_true)
n_classes = y_true_bin.shape[1]

plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue']
class_names = ['Non-Habitable', 'Conservative', 'Optimistic']

for i in range(n_classes):
    # Check if class exists in data
    if i < y_prob.shape[1]: 
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve (Model Generalization)')
plt.legend(loc="lower right")
plt.savefig('nasa_roc_curve.png') # Save for mentor
plt.show()

print("\n✅ Plots saved as 'nasa_confusion_matrix.png' and 'nasa_roc_curve.png'")