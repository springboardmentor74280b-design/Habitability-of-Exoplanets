import joblib
import pandas as pd
import numpy as np
import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models_saved')

# Global Artifacts
model = None
scaler = None
freq_maps = None
ohe_encoder = None
feature_names = None
cols_to_scale = None
freq_encoded_cols = None

def load_artifacts():
    global model, scaler, freq_maps, ohe_encoder, feature_names, cols_to_scale, freq_encoded_cols
    
    if model is None:
        try:
            print("Loading artifacts...")
            model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            freq_maps = joblib.load(os.path.join(MODEL_DIR, 'freq_maps.pkl'))
            ohe_encoder = joblib.load(os.path.join(MODEL_DIR, 'ohe_encoder.pkl'))
            
            with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'r') as f:
                feature_names = json.load(f)
                
            with open(os.path.join(MODEL_DIR, 'cols_to_scale.json'), 'r') as f:
                cols_to_scale = json.load(f)
                
            with open(os.path.join(MODEL_DIR, 'freq_encoded_cols.json'), 'r') as f:
                freq_encoded_cols = json.load(f)
                
            print("Artifacts loaded.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            # Do not crash app, but predictions will fail
            pass

def preprocess_input(df_input):
    """
    Applies the exact same preprocessing as the training pipeline.
    Expects df_input to have the raw columns (or as many as possible).
    """
    # 1. Frequency Encoding
    for col, mapping in freq_maps.items():
        if col in df_input.columns:
            df_input[col] = df_input[col].map(mapping).fillna(0)
    
    # 2. One-Hot Encoding
    if ohe_encoder:
        ohe_cols = ohe_encoder.feature_names_in_
        try:
            ohe_part = df_input[ohe_cols]
            ohe_feats = ohe_encoder.transform(ohe_part)
            ohe_df = pd.DataFrame(ohe_feats, columns=ohe_encoder.get_feature_names_out(ohe_cols), index=df_input.index)
            
            df_input = pd.concat([df_input.drop(columns=ohe_cols), ohe_df], axis=1)
        except Exception as e:
            print(f"OHE Error: {e}")
            
    # 3. Scaling
    # We scale 'cols_to_scale'.
    if cols_to_scale and scaler:
        # We need to fill missing columns with the MEAN from the scaler
        # instead of 0.0. 0.0 is "Out of Distribution" for many positive features.
        # scaler.mean_ corresponds to cols_to_scale order.
        
        for i, c in enumerate(cols_to_scale):
            if c not in df_input.columns:
                # Use mean if available, else 0
                if hasattr(scaler, 'mean_') and i < len(scaler.mean_):
                    df_input[c] = scaler.mean_[i]
                else:
                    df_input[c] = 0.0

        # Now transform
        # Ensure all columns exist before transform (they should now)
        # Handle case where transform input must be 2D
        df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])
        
    # 4. Final Feature Alignment
    # Ensure dataframe matches 'feature_names' exactly
    # Add missing cols as 0 (Categorical OHE/FreqEncoded mostly handled)
    if feature_names:
        for f in feature_names:
            if f not in df_input.columns:
                df_input[f] = 0.0
        
        # Reorder AND subset (discard extras)
        df_final = df_input[feature_names]
        return df_final
    
    return df_input

def predict_single(data_dict):
    """
    data_dict: dict of {feature_name: value}
    """
    if model is None:
        load_artifacts()
        
    # Create DataFrame from input
    df = pd.DataFrame([data_dict])
    
    # -----------------------------------------
    # FEATURE ENGINEERING / PHYSICS DERIVATION
    # -----------------------------------------
    # The model expects many features (P_TEMP_EQUIL, S_LUMINOSITY, P_DENSITY, etc.)
    # If these are 0, prediction fails. We derive them from basic inputs ($M, R, F, T, D$).
    
    # 1. Star Luminosity (S_LUMINOSITY) ~ R^2 * T^4
    s_rad = df.get('S_RADIUS', 0).iloc[0]
    s_temp = df.get('S_TEMPERATURE', 0).iloc[0]
    
    if s_rad > 0 and s_temp > 0:
        l_sun = (s_rad ** 2) * ((s_temp / 5778) ** 4)
        df['S_LUMINOSITY'] = l_sun
        df['S_LOG_LUM'] = np.log10(l_sun) if l_sun > 0 else 0
    
    # 2. Equilibrium Temperature (P_TEMP_EQUIL)
    p_dist = df.get('P_SEMI_MAJOR_AXIS', 0).iloc[0]
    
    if s_rad > 0 and s_temp > 0 and p_dist > 0:
        dist_sr = p_dist * 215.032
        albedo = 0.3
        t_eq = s_temp * np.sqrt(s_rad / (2 * dist_sr)) * ((1 - albedo) ** 0.25)
        df['P_TEMP_EQUIL'] = t_eq
        
        if 'P_TEMP_SURF' not in df.columns or df['P_TEMP_SURF'].iloc[0] == 0:
             df['P_TEMP_SURF'] = t_eq + 30 
             
    # 3. Density (P_DENSITY)
    p_mass = df.get('P_MASS', 0).iloc[0]
    p_rad = df.get('P_RADIUS', 0).iloc[0]
    
    if p_mass > 0 and p_rad > 0:
        density_eu = p_mass / (p_rad ** 3)
        df['P_DENSITY'] = density_eu 
        
    # 4. Escape Velocity (P_ESCAPE)
    if p_mass > 0 and p_rad > 0:
         df['P_ESCAPE'] = np.sqrt(p_mass / p_rad)
         df['P_POTENTIAL'] = p_mass / p_rad 

    # 5. Gravity (P_GRAVITY) - If not provided
    if 'P_GRAVITY' not in df.columns or df['P_GRAVITY'].iloc[0] == 0:
        if p_mass > 0 and p_rad > 0:
            df['P_GRAVITY'] = p_mass / (p_rad ** 2)

    # 6. Star Type (S_TYPE) & S_TYPE_TEMP
    # O > 30000, B 10000-30000, A 7500-10000, F 6000-7500, G 5200-6000, K 3700-5200, M < 3700
    if 'S_TYPE' not in df.columns or not df['S_TYPE'].iloc[0]:
        st = 'G' # Default
        if s_temp >= 30000: st = 'O'
        elif s_temp >= 10000: st = 'B'
        elif s_temp >= 7500: st = 'A'
        elif s_temp >= 6000: st = 'F'
        elif s_temp >= 5200: st = 'G'
        elif s_temp >= 3700: st = 'K'
        elif s_temp > 0: st = 'M'
        
        df['S_TYPE'] = st

    # S_TYPE_TEMP seems to be the same category but used for OHE?
    if 'S_TYPE_TEMP' not in df.columns:
        df['S_TYPE_TEMP'] = df['S_TYPE']

    # 7. Detection Method (P_DETECTION) - Default to 'Transit' if missing
    if 'P_DETECTION' not in df.columns:
        df['P_DETECTION'] = 'Transit'
        
    # 8. Mass Origin - Default to 'M-R relationship' (Most common for Kepler)
    if 'P_MASS_ORIGIN' not in df.columns:
        df['P_MASS_ORIGIN'] = 'M-R relationship' 
        
    # 9. Discovery Year (P_YEAR) - Default to current year for new predictions
    if 'P_YEAR' not in df.columns or df['P_YEAR'].iloc[0] == 0:
        df['P_YEAR'] = 2024
        
    # 10. Constellation (S_CONSTELLATION) - Default to 'Cygnus'
    if 'S_CONSTELLATION' not in df.columns:
        df['S_CONSTELLATION'] = 'Cygnus'
        
    # 11. Habitable Zone (Hz)
    l_sun = df.get('S_LUMINOSITY', 0).iloc[0]
    if l_sun > 0:
        sqrt_l = np.sqrt(l_sun)
        # Fill HZ limits if missing
        if 'S_HZ_OPT_MIN' not in df.columns: df['S_HZ_OPT_MIN'] = 0.75 * sqrt_l
        if 'S_HZ_OPT_MAX' not in df.columns: df['S_HZ_OPT_MAX'] = 1.77 * sqrt_l
        if 'S_HZ_CON_MIN' not in df.columns: df['S_HZ_CON_MIN'] = 0.95 * sqrt_l
        if 'S_HZ_CON_MAX' not in df.columns: df['S_HZ_CON_MAX'] = 1.67 * sqrt_l
        if 'S_HZ_CON0_MIN' not in df.columns: df['S_HZ_CON0_MIN'] = 0.95 * sqrt_l
        if 'S_HZ_CON0_MAX' not in df.columns: df['S_HZ_CON0_MAX'] = 1.67 * sqrt_l
        if 'S_HZ_CON1_MIN' not in df.columns: df['S_HZ_CON1_MIN'] = 0.95 * sqrt_l
        if 'S_HZ_CON1_MAX' not in df.columns: df['S_HZ_CON1_MAX'] = 1.67 * sqrt_l
        if 'S_SNOW_LINE' not in df.columns: df['S_SNOW_LINE'] = 2.7 * sqrt_l
        
    # 12. Tidal Lock
    if 'S_TIDAL_LOCK' not in df.columns:
        per = df.get('P_PERIOD', 100).iloc[0]
        df['S_TIDAL_LOCK'] = 1.0 if per < 30 else 0.0
        
    # 13. Star Physics Defaults (Age, Metallicity, LogG)
    # S_LOG_G: log10(g). g = GM/R^2. Solar g ~ 274 m/s^2 (4.44 dex).
    # g_star = g_sun * (M/M_sun) / (R/R_sun)^2
    # log g = 4.44 + log10(M) - 2*log10(R)
    s_mass = df.get('S_MASS', 0).iloc[0]
    s_rad = df.get('S_RADIUS', 0).iloc[0]
    
    if 'S_LOG_G' not in df.columns:
        if s_mass > 0 and s_rad > 0:
             df['S_LOG_G'] = 4.44 + np.log10(s_mass) - 2 * np.log10(s_rad)
        else:
             df['S_LOG_G'] = 4.44 # Default to Sun
             
    if 'S_METALLICITY' not in df.columns:
        # P_MASS_ORIGIN showed that defaults matter. Metallicity 0 is Solar.
        df['S_METALLICITY'] = 0.0
        
    if 'S_AGE' not in df.columns:
        df['S_AGE'] = 4.5 # Solar Age (Gyr)

    # 14. Error Columns - Hacky but necessary if model overfits on them
    # Set to small non-zero value? Or is 0 fine?
    # Diff analysis showed huge diffs for error cols.
    # But usually models ignore errors unless valid.
    # Let's verify generalizability first with just Star Physics fixes.

    # -----------------------------------------
    # Preprocess
    X = preprocess_input(df)
    
    # Predict
    pred_class = model.predict(X)[0]
    
    pred_proba = model.predict_proba(X)[0]
    
    label_map = {0: 'Non-Habitable', 1: 'Potentially Habitable', 2: 'Very Habitable'}
    
    return {
        "class": int(pred_class),
        "label": label_map.get(int(pred_class), "Unknown"),
        "proba": float(max(pred_proba)) 
    }

def process_csv(file_storage):
    """
    Handles CSV upload, mapping, and prediction.
    """
    df = pd.read_csv(file_storage)
    
    # Preprocess
    if model is None:
        load_artifacts()
        
    X = preprocess_input(df.copy())
    
    preds = model.predict(X)
    probs = model.predict_proba(X) # (N, 3)
    
    habitability_scores = probs[:, 2] * 100 # Percentage
    
    df['Habitability Probability (%)'] = habitability_scores.round(2)
    
    final_df = df[df['Habitability Probability (%)'] > 0].copy()
    
    final_df = final_df.sort_values(by='Habitability Probability (%)', ascending=False)
    
    cols_to_keep = ['Habitability Probability (%)']
    if 'P_NAME' in final_df.columns:
        cols_to_keep.insert(0, 'P_NAME')
    elif 'p_name' in final_df.columns:
        cols_to_keep.insert(0, 'p_name')
    elif 'Name' in final_df.columns:
        cols_to_keep.insert(0, 'Name')
        
    final_df = final_df[cols_to_keep]
    
    return final_df

def analyze_csv_data(file_storage):
    """
    Analyzes the uploaded CSV to generate statistics for visualization.
    Returns a dictionary suitable for JSON response.
    """
    file_storage.seek(0)  # Ensure we read from start
    df = pd.read_csv(file_storage)
    
    # Preprocess & Predict
    if model is None:
        load_artifacts()
        
    X = preprocess_input(df.copy())
    preds = model.predict(X)  # Array of 0, 1, 2
    
    # Counts
    counts = {0: 0, 1: 0, 2: 0}
    unique, u_counts = np.unique(preds, return_counts=True)
    for u, c in zip(unique, u_counts):
        counts[int(u)] = int(c)
        
    stats = {
        'counts': {
            'Non-Habitable': counts[0],
            'Potentially Habitable': counts[1],
            'Very Habitable': counts[2]
        }
    }
    
    # Scatter Plot Data: Mass vs Radius
    if 'P_MASS' in df.columns and 'P_RADIUS' in df.columns:
        # Filter out extreme outliers for better visualization if needed, 
        # but for now send raw data where both exist
        mask = df['P_MASS'].notna() & df['P_RADIUS'].notna()
        stats['scatter'] = {
            'mass': df.loc[mask, 'P_MASS'].tolist(),
            'radius': df.loc[mask, 'P_RADIUS'].tolist(),
            'classes': preds[mask].tolist()
        }
        
    # Histogram Data: Distance (Semi-Major Axis)
    if 'P_SEMI_MAJOR_AXIS' in df.columns:
        valid_dist = df['P_SEMI_MAJOR_AXIS'].dropna()
        # Filter extreme values for better histograms (e.g., < 100 AU)
        valid_dist = valid_dist[valid_dist < 100] 
        stats['distance'] = valid_dist.tolist()

    # Histogram Data: Surface Temperature
    if 'P_TEMP_SURF' in df.columns:
         valid_temp = df['P_TEMP_SURF'].dropna()
         stats['temp'] = valid_temp.tolist()
         
    return stats
