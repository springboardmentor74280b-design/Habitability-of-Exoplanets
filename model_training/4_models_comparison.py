import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import StandardScaler
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

# Scale (For LR and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- DEFINE THE 4 CONTENDERS ---

# 1. Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)

# 2. Weighted SVM
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)

# 3. XGBoost (Weighted)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', num_class=3, n_estimators=100, 
    max_depth=6, learning_rate=0.1, random_state=42
)

# 4. SMOTE + Random Forest
smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)), 
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# --- TRAIN & EVALUATE ---
models = {
    "Logistic Regression": lr_model,
    "Weighted SVM": svm_model,
    "XGBoost (Weighted)": xgb_model,
    "SMOTE + RF": smote_pipeline
}

results = []

print("Comparison in progress...")

for name, model in models.items():
    # Handle Input Differences
    if name == "XGBoost (Weighted)":
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)
    elif name == "SMOTE + RF":
        model.fit(X_train, y_train) # SMOTE handles its own scaling/imbalance internally
        y_pred = model.predict(X_test)
    else: # LR and SVM need Scaled Data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    # Calculate Metrics
    recalls = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "Model": name,
        "Recall Class 1 (Conservative)": recalls[1],
        "Recall Class 2 (Optimistic)": recalls[2],
        "Overall F1 Score": f1
    })

# --- PLOTTING ---
res_df = pd.DataFrame(results)
print("\nResults Table:")
print(res_df)

df_melted = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 7))
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="turbo")

plt.title("The Ultimate Showdown: LR vs SVM vs XGBoost vs SMOTE", fontsize=16)
plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
plt.ylim(0, 1.15)
plt.legend(title="Model", loc='upper right')
plt.grid(axis='y', alpha=0.3)

# Add labels
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.2f', padding=3)

plt.tight_layout()
plt.show()