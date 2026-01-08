import pandas as pd
import numpy as np
from web_app import create_app
from web_app.services import load_artifacts, preprocess_input

def debug_comparison():
    # 1. Get Actual Values from CSV for Kepler-1653 b
    df_csv = pd.read_csv("Dataset/hwc.csv")
    planet_row = df_csv[df_csv['P_NAME'].str.contains("Kepler-1653 b", case=False, na=False)].iloc[0]
    
    print("\n=== Actual values from CSV (Kepler-1653 b) ===")
    print(f"P_MASS: {planet_row['P_MASS']}")
    print(f"P_RADIUS: {planet_row['P_RADIUS']}")
    print(f"P_PERIOD: {planet_row['P_PERIOD']}")
    print(f"P_SEMI_MAJOR_AXIS: {planet_row['P_SEMI_MAJOR_AXIS']}")
    print(f"S_TEMPERATURE: {planet_row['S_TEMPERATURE']}")
    print(f"S_RADIUS: {planet_row['S_RADIUS']}")
    print("-" * 30)
    print(f"S_LUMINOSITY (CSV): {planet_row.get('S_LUMINOSITY')}")
    print(f"S_LOG_LUM (CSV): {planet_row.get('S_LOG_LUM')}")
    print(f"P_TEMP_EQUIL (CSV): {planet_row.get('P_TEMP_EQUIL')}")
    print(f"P_DENSITY (CSV): {planet_row.get('P_DENSITY')}")
    print(f"P_ESCAPE (CSV): {planet_row.get('P_ESCAPE')}")
    print(f"P_POTENTIAL (CSV): {planet_row.get('P_POTENTIAL')}")
    print(f"P_GRAVITY (CSV): {planet_row.get('P_GRAVITY')}")
    
    # 2. Simulate Manual Input (using the basic values from above)
    data = {
        'P_MASS': float(planet_row['P_MASS']),
        'P_RADIUS': float(planet_row['P_RADIUS']),
        'P_FLUX': float(planet_row['P_FLUX']), # using CSV flux to be fair
        'P_PERIOD': float(planet_row['P_PERIOD']),
        'P_SEMI_MAJOR_AXIS': float(planet_row['P_SEMI_MAJOR_AXIS']),
        'S_TEMPERATURE': float(planet_row['S_TEMPERATURE']),
        'S_RADIUS': float(planet_row['S_RADIUS']),
        # Intentionally omitting derived fields to test calculation
    }
    
    print("\n=== Calculated Values (services.py logic) ===")
    
    # Replicating logic from services.py
    s_rad = data['S_RADIUS']
    s_temp = data['S_TEMPERATURE']
    p_mass = data['P_MASS']
    p_rad = data['P_RADIUS']
    p_dist = data['P_SEMI_MAJOR_AXIS']
    
    # 1. Luminosity
    l_sun = (s_rad ** 2) * ((s_temp / 5778) ** 4)
    print(f"S_LUMINOSITY (Calc): {l_sun}")
    print(f"S_LOG_LUM (Calc): {np.log10(l_sun)}")
    
    # 2. Temp Equil
    dist_sr = p_dist * 215.032
    albedo = 0.3 # assumption
    t_eq = s_temp * np.sqrt(s_rad / (2 * dist_sr)) * ((1 - albedo) ** 0.25)
    print(f"P_TEMP_EQUIL (Calc): {t_eq}")
    
    # 3. Density
    # services.py uses: density_eu = p_mass / (p_rad ** 3)
    density_calc = p_mass / (p_rad ** 3)
    print(f"P_DENSITY (Calc - Earth Units?): {density_calc}")
    
    # 4. Escape
    esc = np.sqrt(p_mass / p_rad)
    print(f"P_ESCAPE (Calc): {esc}")
    
    # 5. Gravity
    grav = p_mass / (p_rad ** 2)
    print(f"P_GRAVITY (Calc): {grav}")

if __name__ == "__main__":
    debug_comparison()
