import pandas as pd
import numpy as np
from web_app.services import predict_single, load_artifacts, preprocess_input, model

def debug_g_star():
    load_artifacts()
    
    # Kepler-452b Data from CSV
    # P_MASS 5.0, Radius 1.63, Period 384.8, SMA 1.046
    # Star: 5757 K, 1.11 R_sun, Mass 1.04
    data = {
        'P_MASS': 5.0,
        'P_RADIUS': 1.63,
        'P_FLUX': 1.1,
        'P_PERIOD': 384.84,
        'P_SEMI_MAJOR_AXIS': 1.046,
        'S_TEMPERATURE': 5757,
        'S_RADIUS': 1.11,
        'S_MASS': 1.04
    }
    
    print("\n--- Baseline Check (Current Logic) ---")
    res = predict_single(data)
    print(f"Result: {res}")
    
    # Let's inspect the DATAFRAME created inside predict_single (simulated)
    # We copy logic from services.py briefly to see the values
    df = pd.DataFrame([data])
    
    # Physics Mirror
    s_rad = data['S_RADIUS']
    s_temp = data['S_TEMPERATURE']
    l_sun = (s_rad ** 2) * ((s_temp / 5778) ** 4)
    print(f"Derived Luminosity: {l_sun:.4f} (Expected ~1.2)")
    
    sqrt_l = np.sqrt(l_sun)
    print(f"Sqrt L: {sqrt_l:.4f}")
    print(f"HZ Opt Min: {0.75 * sqrt_l:.4f}")
    print(f"HZ Opt Max: {1.77 * sqrt_l:.4f}")
    
    # What if we tweak P_YEAR?
    print("\n--- Tweak Tests ---")
    years = [2009, 2015, 2024]
    for y in years:
        d = data.copy()
        d['P_YEAR'] = y
        # We need to hack predict_single to accept YEAR if it was passed? 
        # Currently predict_single defaults year if missing.
        # But if we pass it in data_dict, it should respect it?
        # Let's check services.py... 
        # yes: "if 'P_YEAR' not in df.columns or ... == 0"
        
        r = predict_single(d)
        print(f"Year {y}: {r['label']} ({r['proba']:.4f})")
        
    # What if we tweak Detection Method?
    methods = ['Transit', 'Radial Velocity']
    for m in methods:
        d = data.copy()
        d['P_DETECTION'] = m
        r = predict_single(d)
        print(f"Method {m}: {r['label']} ({r['proba']:.4f})")
        
    # What if we tweak HZ boundaries?
    # Maybe our boundaries are too strict?
    # services.py calculates them. 
    # If we pass them explicitly, they override.
    
    # Try wider HZ
    d = data.copy()
    d['S_HZ_OPT_MIN'] = 0.5 * sqrt_l
    d['S_HZ_OPT_MAX'] = 2.5 * sqrt_l
    r = predict_single(d)
    print(f"Wider HZ: {r['label']} ({r['proba']:.4f})")

if __name__ == "__main__":
    debug_g_star()
