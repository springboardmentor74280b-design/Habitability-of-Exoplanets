import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer

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

# 3. Train the CHAMPION XGBoost Model
print("Training Final Champion Model...")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# We use 'softprob' to get the actual probability score (0% to 100%)
model = xgb.XGBClassifier(
    objective='multi:softprob', 
    num_class=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# 4. Generate Rankings (The "Habitability Score")
# We predict probabilities for the Test Set
probs = model.predict_proba(X_test)

# The score is the probability of being Class 1 OR Class 2
# (Column 1 + Column 2)
habitability_scores = probs[:, 1] + probs[:, 2]

# Create a Ranking Table
ranking_df = X_test.copy()
ranking_df['Actual_Class'] = y_test
ranking_df['Habitability_Score'] = habitability_scores
ranking_df['Prediction'] = model.predict(X_test)

# Sort by Score (Highest first)
top_candidates = ranking_df.sort_values(by='Habitability_Score', ascending=False).head(10)

print("\n--- Top 10 Candidate Planets (Ranked by AI Score) ---")
print(top_candidates[['P_MASS_EST', 'P_TEMP_EQUIL', 'Actual_Class', 'Habitability_Score']])

# 5. Save Artifacts for API (Milestone 3)
# We need to save BOTH the imputer (to fill missing values) and the model
with open('final_xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

print("\n✅ Milestone 2 Complete!")
print("1. Champion Model saved as 'final_xgboost_model.pkl'")
print("2. Imputer saved as 'imputer.pkl'")