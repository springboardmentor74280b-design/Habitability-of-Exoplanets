import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load Data
df = pd.read_csv('phl_exoplanet_catalog.csv')

feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]
target_col = 'P_HABITABLE'

df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(df_clean[feature_cols]), columns=feature_cols)
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- CONTENDER 1: Weighted XGBoost (The Current Champion) ---
# Strategy: Raw data + Heavy penalties for mistakes
weights = compute_sample_weight(class_weight='balanced', y=y_train)

model_weighted = xgb.XGBClassifier(
    objective='multi:softmax', num_class=3, n_estimators=100, 
    max_depth=6, learning_rate=0.1, random_state=42
)
print("Training Weighted XGBoost...")
model_weighted.fit(X_train, y_train, sample_weight=weights)


# --- CONTENDER 2: SMOTE + XGBoost (The Hybrid Challenger) ---
# Strategy: Synthetic data + Normal XGBoost (No weights needed!)
# We use a Pipeline to ensure SMOTE only happens on Train data
pipeline_smote = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)), 
    ('xgb', xgb.XGBClassifier(
        objective='multi:softmax', num_class=3, n_estimators=100, 
        max_depth=6, learning_rate=0.1, random_state=42
    ))
])
print("Training SMOTE + XGBoost...")
pipeline_smote.fit(X_train, y_train)


# --- EVALUATION ---
models = {
    "Weighted XGBoost": model_weighted,
    "SMOTE + XGBoost": pipeline_smote
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    
    recalls = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Count False Positives (Non-Habitable classified as Habitable)
    cm = confusion_matrix(y_test, y_pred)
    # Row 0 is Actual Non-Hab. Col 1 and 2 are predicted Habitable.
    false_positives = cm[0, 1] + cm[0, 2]
    
    results.append({
        "Model": name,
        "Class 1 Recall (Conservative)": recalls[1],
        "Class 2 Recall (Optimistic)": recalls[2],
        "False Positives (Noise)": false_positives,
        "Overall F1": f1
    })

# --- PLOTTING ---
res_df = pd.DataFrame(results)
print("\nResults Summary:")
print(res_df)

# Plot Side-by-Side
df_melted = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="coolwarm")
plt.title(" Weighted vs. SMOTE XGBoost", fontsize=14)
plt.ylabel("Score / Count")
plt.ylim(0, 1.2) # Adjusted for seeing "1.0" clearly
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='upper right')

# Add labels
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.2f', padding=3)

plt.show()