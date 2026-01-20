import pandas as pd
import numpy as np
import joblib
import os

def load_model():
    """Loads the pre-trained AI model from disk."""
    filename = 'exohab_model.joblib'
    if os.path.exists(filename):
        print(f"📂 Loading model from {filename}...")
        return joblib.load(filename)
    else:
        print(f"❌ ERROR: {filename} not found! Run train_model.py first.")
        return None

def apply_physics_engine(df):
    """
    The Triple Physics Engine + Universal NASA Translator.
    Now supports Kepler (KOI), TESS, and Archive naming conventions.
    """
    
    # 1. NORMALIZE COLUMNS (Strip whitespace and lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # 2. UNIVERSAL MAPPING
    mapping = {
        # Identity Mappings
        'p_name': 'P_NAME', 'pl_name': 'P_NAME', 'planet name': 'P_NAME',
        'p_mass_est': 'P_MASS_EST', 'pl_bmasse': 'P_MASS_EST', 'mass': 'P_MASS_EST',
        'p_radius_est': 'P_RADIUS_EST', 'pl_rade': 'P_RADIUS_EST', 'radius': 'P_RADIUS_EST',
        'p_period': 'P_PERIOD', 'pl_orbper': 'P_PERIOD', 'orbital period days': 'P_PERIOD',
        'p_distance': 'P_DISTANCE', 'pl_orbsmax': 'P_DISTANCE', 'orbit semi-major axis': 'P_DISTANCE',
        'p_temp_equil': 'P_TEMP_EQUIL', 'pl_eqt': 'P_TEMP_EQUIL', 'equilibrium temperature': 'P_TEMP_EQUIL',
        
        # --- CRITICAL FIX: Map the Target Variable ---
        'p_habitable': 'P_HABITABLE', 'habitable': 'P_HABITABLE', 'phl_habitable': 'P_HABITABLE',

        # Star Parameter Aliases
        's_teff': 'S_TEMPERATURE', 'st_teff': 'S_TEMPERATURE', 'stellar effective temperature': 'S_TEMPERATURE',
        's_temperature': 'S_TEMPERATURE', 's_temp': 'S_TEMPERATURE', 'teff': 'S_TEMPERATURE',
        's_luminosity': 'S_LUMINOSITY', 'st_lum': 'S_LUMINOSITY',
        's_radius': 'S_RADIUS', 'st_rad': 'S_RADIUS', 'stellar radius': 'S_RADIUS',
        's_mass': 'S_MASS', 'st_mass': 'S_MASS', 'stellar mass': 'S_MASS',
        's_metallicity': 'S_METALLICITY', 'st_met': 'S_METALLICITY', 'stellar metallicity': 'S_METALLICITY',

        # NASA Kepler (KOI) Specifics
        'kepoi_name': 'P_NAME_KOI', 'kepler_name': 'P_NAME_KEPLER',
        'koi_period': 'P_PERIOD', 'koi_prad': 'P_RADIUS_EST',
        'koi_teq': 'P_TEMP_EQUIL', 'koi_steff': 'S_TEMPERATURE',
        'koi_srad': 'S_RADIUS', 'koi_smass': 'S_MASS',
        'koi_smet': 'S_METALLICITY', 'koi_dor': 'P_DISTANCE'
    }
    df.rename(columns=mapping, inplace=True)

    # 3. HANDLE DUPLICATES
    df = df.loc[:, ~df.columns.duplicated()]

    # 4. NAME FIX (Kepler vs KOI)
    if 'P_NAME_KEPLER' in df.columns:
        df['P_NAME'] = df['P_NAME_KEPLER'].fillna(df.get('P_NAME_KOI', "Unknown"))
    elif 'P_NAME_KOI' in df.columns and 'P_NAME' not in df.columns:
        df['P_NAME'] = df['P_NAME_KOI']

    # 5. ENSURE NUMERIC TYPES
    cols = ['P_MASS_EST', 'P_RADIUS_EST', 'P_DISTANCE', 'S_RADIUS', 'S_TEMPERATURE', 
            'P_PERIOD', 'P_TEMP_EQUIL', 'S_LUMINOSITY', 'S_MASS', 'S_METALLICITY']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- PHYSICS LOGIC ---

    # A. Mass <-> Radius Logic
    if 'P_RADIUS_EST' in df.columns and 'P_MASS_EST' in df.columns:
        mask_r = df['P_RADIUS_EST'].isna() & df['P_MASS_EST'].notna()
        df.loc[mask_r, 'P_RADIUS_EST'] = df.loc[mask_r, 'P_MASS_EST'].apply(lambda m: m**0.28 if m < 2 else m**0.59)
        
        mask_m = df['P_MASS_EST'].isna() & df['P_RADIUS_EST'].notna()
        df.loc[mask_m, 'P_MASS_EST'] = df.loc[mask_m, 'P_RADIUS_EST'].apply(lambda r: r**3.57 if r < 1.2 else r**1.7)

    # B. Temperature Logic
    if 'P_TEMP_EQUIL' in df.columns:
        mask_t = df['P_TEMP_EQUIL'].isna() & df['S_TEMPERATURE'].notna() & df['S_RADIUS'].notna() & df['P_DISTANCE'].notna()
        if mask_t.any():
            s_rad_au = df.loc[mask_t, 'S_RADIUS'] * 0.00465047 
            df.loc[mask_t, 'P_TEMP_EQUIL'] = df.loc[mask_t, 'S_TEMPERATURE'] * np.sqrt(s_rad_au / (2 * df.loc[mask_t, 'P_DISTANCE']))

    # C. Luminosity Logic
    if 'S_LUMINOSITY' in df.columns:
        mask_l = df['S_LUMINOSITY'].isna() & df['S_RADIUS'].notna() & df['S_TEMPERATURE'].notna()
        if mask_l.any():
            df.loc[mask_l, 'S_LUMINOSITY'] = (df.loc[mask_l, 'S_RADIUS']**2) * ((df.loc[mask_l, 'S_TEMPERATURE'] / 5778)**4)

    return df

def predict_habitability(df, pipeline):
    """Takes clean data, returns predictions. Handles NaN values safely."""
    features = ['P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
                'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 
                'S_RADIUS', 'S_MASS', 'S_METALLICITY']
    
    # Ensure all features exist (fill missing with -99 for the AI)
    for f in features:
        if f not in df.columns:
            df[f] = -99
    
    X = df[features].fillna(-99)
    
    # Predict
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    
    # Format results
    results = []
    classes = {0: "Non-Habitable", 1: "Habitable", 2: "Optimistic"}
    
    for i, p in enumerate(preds):
        # Helper to safely get float values
        def safe_float(val):
            if pd.isna(val) or val == -99:
                return None  # Convert NaN to None (JSON null)
            return float(val)

        results.append({
            "P_NAME": str(df.iloc[i].get('P_NAME', f"Planet_{i}")),
            "prediction": classes.get(p, "Unknown"),
            "probability": float(max(probs[i])),
            "mass": safe_float(df.iloc[i].get('P_MASS_EST')),
            "radius": safe_float(df.iloc[i].get('P_RADIUS_EST')),
            "temp": safe_float(df.iloc[i].get('P_TEMP_EQUIL')),
            "period": safe_float(df.iloc[i].get('P_PERIOD')),
            "distance": safe_float(df.iloc[i].get('P_DISTANCE')),
            "star_temp": safe_float(df.iloc[i].get('S_TEMPERATURE')),
            # ... existing fields ...
            "s_radius": safe_float(df.iloc[i].get('S_RADIUS')),
            "s_mass": safe_float(df.iloc[i].get('S_MASS')),
            "s_metallicity": safe_float(df.iloc[i].get('S_METALLICITY')),
            "star_lum": safe_float(df.iloc[i].get('S_LUMINOSITY'))
        
        })
        
    return results