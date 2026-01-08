import pandas as pd
import numpy as np
from web_app.services import load_artifacts, preprocess_input, model

def debug_diff():
    load_artifacts()
    from web_app.services import model, feature_names
    
    # 1. Gold Standard
    df_csv = pd.read_csv("Dataset/hwc.csv")
    planet_row = df_csv[df_csv['P_NAME'].str.contains("Kepler-452 b", case=False, na=False)].iloc[0]
    df_gold = pd.DataFrame([planet_row])
    X_gold = preprocess_input(df_gold.copy())
    
    # 2. Manual (Recreating what services.py produces)
    # We can actually USE services.py's logic if we extract it, 
    # but let's approximate the RESULT of services.py manually to control it.
    
    manual_data = {
         'P_MASS': float(planet_row['P_MASS']),
         'P_RADIUS': float(planet_row['P_RADIUS']),
         'P_FLUX': float(planet_row['P_FLUX']), 
         'P_PERIOD': float(planet_row['P_PERIOD']),
         'P_SEMI_MAJOR_AXIS': float(planet_row['P_SEMI_MAJOR_AXIS']),
         'S_TEMPERATURE': float(planet_row['S_TEMPERATURE']),
         'S_RADIUS': float(planet_row['S_RADIUS']),
         # Derived Physics
         'S_LUMINOSITY': (float(planet_row['S_RADIUS'])**2) * ((float(planet_row['S_TEMPERATURE'])/5778)**4),
         'P_TEMP_EQUIL': float(planet_row['S_TEMPERATURE']) * np.sqrt(float(planet_row['S_RADIUS']) / (2 * float(planet_row['P_SEMI_MAJOR_AXIS']) * 215.032)) * ((1 - 0.3) ** 0.25),
         'P_DENSITY': float(planet_row['P_MASS']) / (float(planet_row['P_RADIUS'])**3),
         'P_ESCAPE': np.sqrt(float(planet_row['P_MASS']) / float(planet_row['P_RADIUS'])),
         'P_GRAVITY': float(planet_row['P_MASS']) / (float(planet_row['P_RADIUS'])**2),
         # Defaults
         'S_TYPE': 'K', # G, K, etc.
         'S_TYPE_TEMP': 'K',
         'P_DETECTION': 'Transit',
         'P_MASS_ORIGIN': 'Mass'
    }
    # Add Log Lum
    manual_data['S_LOG_LUM'] = np.log10(manual_data['S_LUMINOSITY'])
    
    df_manual = pd.DataFrame([manual_data])
    X_manual = preprocess_input(df_manual.copy())
    
    print(f"Gold Prob: {model.predict_proba(X_gold)[0][2]}")
    print(f"Manual Prob: {model.predict_proba(X_manual)[0][2]}")
    
    # 3. Diff ALL columns
    print("\n--- Significant Differences (> 5%) ---")
    diffs = []
    for col in X_gold.columns:
        val_gold = X_gold[col].iloc[0]
        val_manual = X_manual[col].iloc[0]
        
        # If both 0, skip
        if abs(val_gold) < 1e-6 and abs(val_manual) < 1e-6: continue
        
        # Calculate diff
        diff = abs(val_gold - val_manual)
        avg = (abs(val_gold) + abs(val_manual)) / 2
        pct_diff = (diff / avg) * 100 if avg > 0 else 0
        
        if pct_diff > 5:
            diffs.append((col, val_gold, val_manual, pct_diff))
            
    # Sort by diff magnitude
    diffs.sort(key=lambda x: x[3], reverse=True)
    for col, g, m, p in diffs:
        # Filter out HZ and Year if we already know they are diff (optional)
        print(f"{col}: Gold={g:.4f}, Manual={m:.4f} (Diff: {p:.1f}%)")
            
    # Check if 'P_YEAR' is in features
    if 'P_YEAR' in X_gold.columns:
        print(f"\nP_YEAR in Gold: {X_gold['P_YEAR'].iloc[0]}")
        print(f"P_YEAR in Manual: {X_manual['P_YEAR'].iloc[0]}")

if __name__ == "__main__":
    debug_diff()
