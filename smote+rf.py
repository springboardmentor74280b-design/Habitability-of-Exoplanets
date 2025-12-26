import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load Data
df = pd.read_csv('phl_exoplanet_catalog.csv')

# Features & Cleaning
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]
target_col = 'P_HABITABLE'

df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
imputer = SimpleImputer(strategy='median')
X = df_clean[feature_cols]
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df_clean[target_col]

# 2. Split Data (Stratify is crucial)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. The SMOTE Pipeline
# This automatically generates new 'Green' and 'Blue' dots during training
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)), 
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])

# 4. Train
print("Training Final SMOTE Model...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 5. Final Visualization
print("\nClassification Report (Compare Recall to Baseline!):")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("SMOTE Confusion Matrix\n(SMOTE + Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Save for Milestone 3 (API)
with open('final_smote+rf_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("\n✅ Final Model saved as 'final_smote+rf_model.pkl'")