import pandas as pd
import pickle
import numpy as np

# 1. Load Raw Data
print("Loading Raw Data...")
df = pd.read_csv('phl_exoplanet_catalog.csv')

# Features needed for the model
feature_cols = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 'S_RADIUS', 
    'S_MASS', 'S_METALLICITY'
]

# We need to keep the Planet Name for the leaderboard!
# Assuming the column name is 'P_NAME' or similar. Check your CSV if it fails.
if 'P_NAME' in df.columns:
    name_col = 'P_NAME'
else:
    name_col = df.columns[0] # Fallback to first column

# 2. Load the Unified Pipeline
print("Loading AI Pipeline...")
with open('final_pipeline.pkl', 'rb') as f:
    pipeline_artifacts = pickle.load(f)

# Extract the parts (The pipeline might be a dict or a Pipeline object)
# based on our last script, it was a dict: {'preprocessor': ..., 'model': ...}
if isinstance(pipeline_artifacts, dict):
    preprocessor = pipeline_artifacts["preprocessor"]
    model = pipeline_artifacts["model"]
    
    # Preprocess the FULL dataset
    # Note: We must handle NaNs exactly like the training phase
    X_raw = df[feature_cols]
    X_processed = preprocessor.transform(X_raw)
    
else:
    # If it was saved as a single Pipeline object
    pipeline = pipeline_artifacts
    X_raw = df[feature_cols]
    # We can't easily get intermediate steps if it's one block, 
    # but we can just predict directly if it handles NaNs.
    # For safety, let's assume the dict structure we created in section8_validation.py
    print("Error: Expected dictionary artifact. Please ensure 'section8_validation.py' was run.")
    exit()

# 3. Generate Predictions for the Universe
print("Scoring 4000+ Planets...")
probs = model.predict_proba(X_processed)
preds = model.predict(X_processed)

# 4. Calculate Habitability Score
# Score = Probability(Conservative) + Probability(Optimistic)
habitability_scores = probs[:, 1] + probs[:, 2]

# 5. Create the Leaderboard DataFrame
leaderboard = df[[name_col] + feature_cols].copy()
leaderboard['Predicted_Class'] = preds
leaderboard['Habitability_Score'] = habitability_scores
leaderboard['Confidence_Conservative'] = probs[:, 1]
leaderboard['Confidence_Optimistic'] = probs[:, 2]

# 6. Sort and Save
leaderboard = leaderboard.sort_values(by='Habitability_Score', ascending=False)

# Save to CSV
filename = 'ranked_planets_leaderboard.csv'
leaderboard.to_csv(filename, index=False)

print(f"\n✅ Leaderboard Generated: {filename}")
print(f"Top 5 Planets found by your AI:")
print(leaderboard[[name_col, 'Predicted_Class', 'Habitability_Score']].head(5))