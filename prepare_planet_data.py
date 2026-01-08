"""
Prepare Planet Presets for Dropdown Selection
==============================================

This script creates a curated list of planets including:
- Earth (reference baseline)
- 50 diverse Kepler planets
- 10 test samples

Output: outputs/planet_presets.json
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# Paths
CATALOG_PATH = "phl_exoplanet_catalog_2019.csv"
OUTPUT_PATH = "outputs/planet_presets.json"

# Required features for the model
REQUIRED_FEATURES = [
    'P_MASS_EST', 'P_RADIUS_EST', 'P_TEMP_EQUIL', 
    'P_PERIOD', 'P_FLUX', 'S_MASS', 'S_RADIUS', 'S_TEMPERATURE'
]

def get_earth_data():
    """Return Earth's data as reference baseline."""
    return {
        "name": "Earth",
        "description": "Our home planet - the reference for habitability",
        "data": {
            "P_MASS_EST": 1.0,
            "P_RADIUS_EST": 1.0,
            "P_TEMP_EQUIL": 288.0,
            "P_PERIOD": 365.25,
            "P_FLUX": 1.0,
            "S_MASS": 1.0,
            "S_RADIUS": 1.0,
            "S_TEMP": 5778.0
        }
    }

def select_diverse_kepler_planets(df, n=50):
    """
    Select diverse Kepler planets to showcase variety in:
    - Temperature (hot, warm, cold)
    - Size (small, medium, large)
    - Orbital period (short, medium, long)
    """
    kepler = df[df['P_NAME'].str.contains('Kepler', na=False)].copy()
    
    # Filter planets with all required features
    kepler = kepler.dropna(subset=REQUIRED_FEATURES)
    
    print(f"Total Kepler planets with complete data: {len(kepler)}")
    
    # Create diversity bins
    kepler['temp_bin'] = pd.qcut(kepler['P_TEMP_EQUIL'], q=5, labels=['very_cold', 'cold', 'moderate', 'warm', 'hot'], duplicates='drop')
    kepler['size_bin'] = pd.qcut(kepler['P_RADIUS_EST'], q=3, labels=['small', 'medium', 'large'], duplicates='drop')
    kepler['period_bin'] = pd.qcut(kepler['P_PERIOD'], q=3, labels=['short', 'medium', 'long'], duplicates='drop')
    
    # Sample from each combination to ensure diversity
    diverse_planets = []
    
    # Try to get diverse samples
    for temp in kepler['temp_bin'].unique():
        if pd.isna(temp):
            continue
        temp_group = kepler[kepler['temp_bin'] == temp]
        
        for size in temp_group['size_bin'].unique():
            if pd.isna(size):
                continue
            size_group = temp_group[temp_group['size_bin'] == size]
            
            # Sample 1-2 from each group
            sample_size = min(2, len(size_group))
            if sample_size > 0:
                samples = size_group.sample(n=sample_size, random_state=42)
                diverse_planets.append(samples)
    
    # Combine all diverse samples
    if diverse_planets:
        result = pd.concat(diverse_planets, ignore_index=True)
    else:
        result = kepler.sample(n=min(n, len(kepler)), random_state=42)
    
    # If we don't have enough, add more random samples
    if len(result) < n:
        remaining = kepler[~kepler.index.isin(result.index)]
        additional = remaining.sample(n=min(n - len(result), len(remaining)), random_state=42)
        result = pd.concat([result, additional], ignore_index=True)
    
    # Limit to n planets
    result = result.head(n)
    
    # Sort by temperature for easier browsing
    result = result.sort_values('P_TEMP_EQUIL', ascending=False)
    
    print(f"Selected {len(result)} diverse Kepler planets")
    
    # Convert to list of dictionaries
    planets = []
    for _, row in result.iterrows():
        planet = {
            "name": row['P_NAME'],
            "description": f"Temp: {row['P_TEMP_EQUIL']:.0f}K, Radius: {row['P_RADIUS_EST']:.2f}R⊕, Period: {row['P_PERIOD']:.1f}d",
            "data": {
                "P_MASS_EST": float(row['P_MASS_EST']),
                "P_RADIUS_EST": float(row['P_RADIUS_EST']),
                "P_TEMP_EQUIL": float(row['P_TEMP_EQUIL']),
                "P_PERIOD": float(row['P_PERIOD']),
                "P_FLUX": float(row['P_FLUX']),
                "S_MASS": float(row['S_MASS']),
                "S_RADIUS": float(row['S_RADIUS']),
                "S_TEMP": float(row['S_TEMPERATURE'])
            }
        }
        planets.append(planet)
    
    return planets

def get_test_samples(df, n=10):
    """Get representative test samples from the dataset."""
    # Filter planets with all required features
    valid = df.dropna(subset=REQUIRED_FEATURES)
    
    # Select diverse samples
    samples = valid.sample(n=min(n, len(valid)), random_state=42)
    
    print(f"Selected {len(samples)} test samples")
    
    # Convert to list of dictionaries
    test_data = []
    for idx, row in enumerate(samples.iterrows(), 1):
        _, row = row
        sample = {
            "name": f"Test Sample {idx}",
            "description": f"{row.get('P_NAME', 'Unknown')} - Temp: {row['P_TEMP_EQUIL']:.0f}K",
            "data": {
                "P_MASS_EST": float(row['P_MASS_EST']),
                "P_RADIUS_EST": float(row['P_RADIUS_EST']),
                "P_TEMP_EQUIL": float(row['P_TEMP_EQUIL']),
                "P_PERIOD": float(row['P_PERIOD']),
                "P_FLUX": float(row['P_FLUX']),
                "S_MASS": float(row['S_MASS']),
                "S_RADIUS": float(row['S_RADIUS']),
                "S_TEMP": float(row['S_TEMPERATURE'])
            }
        }
        test_data.append(sample)
    
    return test_data

def main():
    """Main function to prepare planet presets."""
    print("=" * 80)
    print("PREPARING PLANET PRESETS")
    print("=" * 80)
    
    # Load catalog
    print(f"\nLoading catalog from: {CATALOG_PATH}")
    df = pd.read_csv(CATALOG_PATH)
    print(f"Total planets in catalog: {len(df)}")
    
    # Prepare data
    print("\n" + "-" * 80)
    print("1. Adding Earth reference data...")
    earth = get_earth_data()
    
    print("\n" + "-" * 80)
    print("2. Selecting diverse Kepler planets...")
    kepler_planets = select_diverse_kepler_planets(df, n=50)
    
    print("\n" + "-" * 80)
    print("3. Getting test samples...")
    test_samples = get_test_samples(df, n=10)
    
    # Create output structure
    output = {
        "earth": earth,
        "kepler": kepler_planets,
        "test_samples": test_samples,
        "metadata": {
            "total_planets": len(kepler_planets) + len(test_samples) + 1,
            "kepler_count": len(kepler_planets),
            "test_samples_count": len(test_samples),
            "required_features": REQUIRED_FEATURES
        }
    }
    
    # Save to file
    print("\n" + "-" * 80)
    print(f"Saving to: {OUTPUT_PATH}")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved {output['metadata']['total_planets']} planets to {OUTPUT_PATH}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Earth:         1 planet")
    print(f"Kepler:        {len(kepler_planets)} planets")
    print(f"Test Samples:  {len(test_samples)} samples")
    print(f"Total:         {output['metadata']['total_planets']} planets")
    print("=" * 80)
    
    return output

if __name__ == "__main__":
    main()
