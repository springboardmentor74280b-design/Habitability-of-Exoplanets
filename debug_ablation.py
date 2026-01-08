import pandas as pd
import numpy as np
import joblib
import json
import os
from web_app.services import load_artifacts, preprocess_input, model

def debug_ablation():
    # Load Artifacts
    load_artifacts()
    from web_app.services import model
    
    # 1. Get Actual Values from CSV for Kepler-1653 b (The Gold Standard)
    df_csv = pd.read_csv("Dataset/hwc.csv")
    planet_row = df_csv[df_csv['P_NAME'].str.contains("Kepler-1653 b", case=False, na=False)].iloc[0]
    
    # Create a single-row DataFrame
    df_gold = pd.DataFrame([planet_row])
    
    # Preprocess and Predict Gold Standard
    # Note: process_csv does a copy(), let's mimic services.py processing
    # BUT services.py process_csv calls preprocess_input on the whole DF.
    # preprocess_input handles frequency encoding mapping.
    
    print("\n--- 1. Gold Standard (CSV Row) ---")
    X_gold = preprocess_input(df_gold.copy())
    prob_gold = model.predict_proba(X_gold)[0][2] # Class 2
    print(f"Probability (Habitable): {prob_gold:.4f}")
    
    # 2. Simulate Manual Input by stripping away features
    print("\n--- 2. Ablation Study (Removing features one by one) ---")
    
    mod_df = df_gold.copy()
    
    # List of features we DON'T ask for in the form
    features_missing_in_form = [
        'S_CONSTELLATION',
        'P_YEAR',
        'P_MASS_ORIGIN',
        'S_NAME', # We derived derived S_LUMINOSITY etc, but S_NAME itself?
        'P_UPDATED'
    ]
    
    # Baseline Manual (with physics fix)
    # This simulates what we strictly send from the form + physics fix
    manual_data = {
         'P_MASS': float(planet_row['P_MASS']),
         'P_RADIUS': float(planet_row['P_RADIUS']),
         'P_FLUX': float(planet_row['P_FLUX']), 
         'P_PERIOD': float(planet_row['P_PERIOD']),
         'P_SEMI_MAJOR_AXIS': float(planet_row['P_SEMI_MAJOR_AXIS']),
         'S_TEMPERATURE': float(planet_row['S_TEMPERATURE']),
         'S_RADIUS': float(planet_row['S_RADIUS']),
         # Physics derivations (Simulated)
         'S_LUMINOSITY': (float(planet_row['S_RADIUS'])**2) * ((float(planet_row['S_TEMPERATURE'])/5778)**4),
         # ... Assume physics is perfect for now, we validated that.
         # P_DETECTION & S_TYPE defaults
         'S_TYPE': 'K', # Derived
         'P_DETECTION': 'Transit' # Default
    }
    
    # Now, let's verify what happens if we pass ONLY this manual-like dict
    df_manual = pd.DataFrame([manual_data])
    
    # We need to manually add the columns that preprocess_input EXPECTS to exist (to initiate freq map loops)
    # If they don't exist in manual DF, preprocess_input loop checks `if col in df.columns`.
    # So they are skipped -> 0.
    
    X_manual = preprocess_input(df_manual.copy())
    prob_manual = model.predict_proba(X_manual)[0][2]
    print(f"Manual Input (Simulated Physics + Derived S_TYPE): {prob_manual:.4f}")
    
    # 3. Add back features to Manual to see which one restores the score
    print("\n--- 3. Restoration Study (Adding back features to Manual) ---")
    
    features_to_test = ['S_CONSTELLATION', 'P_YEAR', 'P_MASS_ORIGIN', 'S_LOG_G', 'S_METALLICITY', 'S_AGE']
    
    for feat in features_to_test:
        df_test = df_manual.copy()
        val = planet_row.get(feat, 0)
        df_test[feat] = val
        
        X_test = preprocess_input(df_test)
        prob = model.predict_proba(X_test)[0][2]
        print(f"Adding {feat} ({val}): {prob:.4f}")
        
    # Test combination
    print("\n--- 4. All Missing Added ---")
    df_all = df_manual.copy()
    for feat in features_to_test:
        df_all[feat] = planet_row.get(feat, 0)
    X_all = preprocess_input(df_all)
    prob_all = model.predict_proba(X_all)[0][2]
    print(f"All Added: {prob_all:.4f}")

if __name__ == "__main__":
    debug_ablation()
