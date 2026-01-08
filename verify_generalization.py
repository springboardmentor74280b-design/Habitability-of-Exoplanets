import pandas as pd
import numpy as np
from web_app.services import predict_single, load_artifacts, preprocess_input, model

def verify_generalization():
    load_artifacts()
    
    # Planets to test (Name, Expected Class)
    # Class 2 = Very Habitable, Class 1 = Potentially, Class 0 = Non
    test_cases = [
        ("Kepler-1653 b", 2), # The one we fixed (K-type)
        ("Kepler-452 b", 2),  # Earth cousin (G-type)
        ("Kepler-62 f", 2),   # Super-Earth (K-type)
        ("Teegarden's Star b", 2), # Red Dwarf (M-type) - tight orbit
        ("TRAPPIST-1 e", 2),  # Red Dwarf (M-type)
        ("Earth", 2)          # Control
    ]
    
    df_csv = pd.read_csv("Dataset/hwc.csv")
    
    print(f"{'Planet':<20} | {'Type':<5} | {'Exp':<3} | {'Pred':<3} | {'Prob':<6} | {'Status'}")
    print("-" * 65)
    
    for planet_name, expected_class in test_cases:
        # 1. Get Real Data
        # Fuzzy match name
        rows = df_csv[df_csv['P_NAME'].str.contains(planet_name, case=False, na=False)]
        if rows.empty:
            print(f"{planet_name:<20} | ???   | {expected_class}   | N/A | N/A    | Not in CSV")
            continue
        
        row = rows.iloc[0]
        
        # 2. Extract ONLY the fields the user can type in the form
        manual_input = {
            'P_MASS': float(row['P_MASS']),
            'P_RADIUS': float(row['P_RADIUS']),
            'P_FLUX': float(row['P_FLUX']),
            'P_PERIOD': float(row['P_PERIOD']),
            'P_SEMI_MAJOR_AXIS': float(row['P_SEMI_MAJOR_AXIS']),
            'S_TEMPERATURE': float(row['S_TEMPERATURE']),
            'S_RADIUS': float(row['S_RADIUS']),
            'S_MASS': float(row['S_MASS']),
            # P_GRAVITY is optional in form, but let's assume user doesn't calculate it
            # so we leave it out to test our derivation
        }
        
        # 3. Predict using our services.py logic
        result = predict_single(manual_input)
        
        # 4. Compare
        pred_class = result['class']
        prob = result['proba']
        star_type = 'M' if row['S_TEMPERATURE'] < 3700 else 'K' if row['S_TEMPERATURE'] < 5200 else 'G' if row['S_TEMPERATURE'] < 6000 else 'F'
        
        status = "PASS" if pred_class == expected_class else "FAIL"
        # Allow Class 1 for some debatable ones
        if expected_class == 2 and pred_class == 1: status = "WARN (1)"
        
        print(f"{planet_name:<20} | {star_type:<5} | {expected_class:<3} | {pred_class:<3} | {prob:.4f} | {status}")

if __name__ == "__main__":
    verify_generalization()
