import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
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

# Clean
df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(df_clean[feature_cols]), columns=feature_cols)
y = df_clean[target_col]

# 2. Split Data (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define Pipeline: SMOTE + XGBoost
# Note: We do NOT use 'sample_weight' here because SMOTE balances the data for us.
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)), 
    ('xgb', xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=3, 
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42
    ))
])

# 4. Train
print("Training SMOTE + XGBoost...")
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)

print("\n--- SMOTE + XGBoost Report ---")
print(classification_report(y_test, y_pred))

# 6. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')

plt.title("SMOTE + XGBoost Confusion Matrix\n(Hybrid Approach)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0.5, 1.5, 2.5], ['0: Non-Hab', '1: Conservative', '2: Optimistic'])
plt.yticks([0.5, 1.5, 2.5], ['0: Non-Hab', '1: Conservative', '2: Optimistic'])

plt.tight_layout()
plt.show()
print("✅ Plot Generated.")