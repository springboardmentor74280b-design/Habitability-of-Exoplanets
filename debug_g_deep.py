import pandas as pd
import numpy as np
from web_app.services import predict_single, load_artifacts, preprocess_input, model

def debug_g_star_deep():
    load_artifacts()
    
    # 1. Gold (CSV)
    df_csv = pd.read_csv("Dataset/hwc.csv")
    gold_row = df_csv[df_csv['P_NAME'].str.contains("Kepler-452 b", case=False, na=False)].iloc[0]
    
    # 2. Manual (Our Best Calculation)
    manual_input = {
        'P_MASS': float(gold_row['P_MASS']),
        'P_RADIUS': float(gold_row['P_RADIUS']),
        'P_FLUX': float(gold_row['P_FLUX']),
        'P_PERIOD': float(gold_row['P_PERIOD']),
        'P_SEMI_MAJOR_AXIS': float(gold_row['P_SEMI_MAJOR_AXIS']),
        'S_TEMPERATURE': float(gold_row['S_TEMPERATURE']),
        'S_RADIUS': float(gold_row['S_RADIUS']),
        'S_MASS': float(gold_row['S_MASS']),
        # We let services.py derive the rest
    }
    
    print("\n--- Feature Swapping ---")
    
    # We want to see which derived feature is WRONG.
    # So we will run predict_single, but we will "inject" Gold values for derived features
    # into the input dict to override services.py derivation.
    
    derived_cols = [
        'S_LUMINOSITY', 'S_LOG_LUM', 
        'P_TEMP_EQUIL', 'P_DENSITY', 'P_ESCAPE', 'P_POTENTIAL', 'P_GRAVITY',
        'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'S_HZ_CON_MIN', 'S_HZ_CON_MAX',
        'S_HZ_CON0_MIN', 'S_HZ_CON0_MAX', 'S_HZ_CON1_MIN', 'S_HZ_CON1_MAX',
        'S_SNOW_LINE', 'S_ABIO_ZONE', 'S_TIDAL_LOCK',
        'P_YEAR', 'S_CONSTELLATION', 'S_TYPE', 'P_DETECTION', 'P_MASS_ORIGIN'
    ]
    
    # Baseline
    res = predict_single(manual_input)
    print(f"Baseline Manual: {res['label']} ({res['proba']:.4f})")
    
    for col in derived_cols:
        d = manual_input.copy()
        gold_val = gold_row.get(col)
        # Handle NaN/missing
        if pd.isna(gold_val): continue
        
        d[col] = gold_val # Override derivation
        res = predict_single(d)
        
        # Check if score improved significantly (Class 2 prob)
        # But predict_single returns max prob logic.
        # We need raw prob of class 2.
        # Let's trust predict_single output for now.
        if res['class'] == 2 and res['proba'] > 0.5:
             print(f"*** Found It? Swapping {col} (Val: {gold_val}) -> HABITABLE ({res['proba']})")
        else:
             print(f"Swapping {col}: {res['label']} ({res['proba']:.4f})")
             
    # Test ALL together
    print("\n--- Swapping ALL Derived Features ---")
    d_all = manual_input.copy()
    for col in derived_cols:
        val = gold_row.get(col)
        if not pd.isna(val):
            d_all[col] = val
    res_all = predict_single(d_all)
    print(f"All Derived Swapped: {res_all['label']} ({res_all['proba']:.4f})")

if __name__ == "__main__":
    debug_g_star_deep()
