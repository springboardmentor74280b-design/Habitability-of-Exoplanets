import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the Full Catalog
print("Loading the entire Exoplanet Catalog...")
df = pd.read_csv('phl_exoplanet_catalog.csv')

# Define features (Must match training exactly)
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]
target_col = 'P_HABITABLE'

# We keep the 'P_NAME' column so we can print the names later!
cols_to_keep = feature_cols + [target_col, 'P_NAME']
df_clean = df[cols_to_keep].dropna(subset=feature_cols + [target_col])

X_full = df_clean[feature_cols]
y_full = df_clean[target_col]

# 2. Load the Champion Brain
print("Loading SMOTE + XGBoost Model...")
with open('final_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# 3. Predict on the Full Universe
print("Scanning 4000+ planets for life...")
y_pred = model_pipeline.predict(X_full)

# 4. Results
print("\n--- REPORT ON FULL DATASET ---")
print(classification_report(y_full, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_full, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Full Universe Classification\n(Includes Training Data)")
plt.xlabel("AI Prediction")
plt.ylabel("Actual Label")
plt.show()

# 5. The "Roll Call" - Which planets did it choose?
df_clean['AI_PREDICTION'] = y_pred

print("\n--- AI's Favorite Planets (Class 1: Conservative) ---")
class_1_planets = df_clean[df_clean['AI_PREDICTION'] == 1]['P_NAME'].values
print(f"Count: {len(class_1_planets)}")
print(class_1_planets)

print("\n--- AI's Optimistic Picks (Class 2: Optimistic) ---")
class_2_planets = df_clean[df_clean['AI_PREDICTION'] == 2]['P_NAME'].values
print(f"Count: {len(class_2_planets)}")
print(class_2_planets)

# 6. Check for "Hallucinations" (False Positives)
# Did it flag any Dead Planet (0) as Habitable (1 or 2)?
mistakes = df_clean[ (df_clean[target_col] == 0) & (df_clean['AI_PREDICTION'] > 0) ]

if len(mistakes) > 0:
    print(f"\n⚠️ HALLUCINATIONS DETECTED: {len(mistakes)}")
    print("The AI incorrectly thinks these dead planets are alive:")
    print(mistakes[['P_NAME', 'P_HABITABLE', 'AI_PREDICTION']])
else:
    print("\n✅ ZERO HALLUCINATIONS.")
    print("The AI did not mistakenly flag any dead rocks as habitable.")