# debug_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def debug_model():
    """Debug the model to understand why Earth is non-habitable"""
    print("🔍 Debugging Model Predictions...")
    
    # Load everything
    model = joblib.load("saved_models/xgb_habitability_model.pkl")
    preprocessor = joblib.load("saved_models/preprocessor.pkl")
    feature_columns = joblib.load("saved_models/feature_columns.pkl")
    
    print(f"📊 Model has {len(feature_columns)} features")
    
    # Test different scenarios
    test_cases = [
        {
            'name': 'Earth (Perfect)',
            'P_RADIUS': 1.0,
            'P_MASS': 1.0,
            'P_GRAVITY': 1.0,
            'P_TEMP_EQUIL': 288.0,
            'P_DENSITY': 5.51,
            'P_ORBPER': 365.25
        },
        {
            'name': 'Earth (Slightly Warmer)',
            'P_RADIUS': 1.0,
            'P_MASS': 1.0,
            'P_GRAVITY': 1.0,
            'P_TEMP_EQUIL': 300.0,
            'P_DENSITY': 5.51,
            'P_ORBPER': 365.25
        },
        {
            'name': 'Super Earth',
            'P_RADIUS': 1.5,
            'P_MASS': 5.0,
            'P_GRAVITY': 2.2,
            'P_TEMP_EQUIL': 288.0,
            'P_DENSITY': 6.0,
            'P_ORBPER': 200.0
        },
        {
            'name': 'Mars-like',
            'P_RADIUS': 0.53,
            'P_MASS': 0.11,
            'P_GRAVITY': 0.38,
            'P_TEMP_EQUIL': 210.0,
            'P_DENSITY': 3.93,
            'P_ORBPER': 687.0
        }
    ]
    
    for test in test_cases:
        print(f"\n🌍 Testing: {test['name']}")
        print(f"  Parameters: Radius={test['P_RADIUS']}, Mass={test['P_MASS']}, "
              f"Temp={test['P_TEMP_EQUIL']}K")
        
        # Create input with all features
        input_dict = {}
        for feature in feature_columns:
            # Set defaults
            if feature.startswith('P_') and '_ERROR' not in feature:
                if 'RADIUS' in feature:
                    input_dict[feature] = test['P_RADIUS']
                elif 'MASS' in feature:
                    input_dict[feature] = test['P_MASS']
                elif 'GRAV' in feature:
                    input_dict[feature] = test['P_GRAVITY']
                elif 'TEMP' in feature:
                    input_dict[feature] = test['P_TEMP_EQUIL']
                elif 'DENS' in feature:
                    input_dict[feature] = test['P_DENSITY']
                elif 'PERIOD' in feature or 'ORB' in feature:
                    input_dict[feature] = test['P_ORBPER']
                else:
                    input_dict[feature] = 0.0
            elif feature.startswith('S_'):
                if 'LUM' in feature:
                    input_dict[feature] = 1.0
                else:
                    input_dict[feature] = 0.0
            else:
                input_dict[feature] = 0.0
        
        df = pd.DataFrame([input_dict])
        df = df[feature_columns]
        
        # Transform and predict
        processed = preprocessor.transform(df)
        prediction = model.predict(processed)[0]
        probabilities = model.predict_proba(processed)[0]
        
        print(f"  Prediction: {prediction} "
              f"(0=Non-Habitable, 1=Potentially Habitable, 2=Highly Habitable)")
        print(f"  Probabilities: Non={probabilities[0]*100:.2f}%, "
              f"Potential={probabilities[1]*100:.2f}%, "
              f"High={probabilities[2]*100:.2f}%")
        
        # Check feature importance if prediction seems wrong
        if test['name'] == 'Earth (Perfect)' and prediction == 0:
            print(f"  ⚠️  Earth predicted as Non-Habitable!")
            print(f"  This suggests either:")
            print(f"  1. Training data has mislabeled Earth-like planets")
            print(f"  2. Missing important features in our test input")
            print(f"  3. Model learned unexpected patterns")

def check_training_data():
    """Check the original training data"""
    print("\n📊 Checking Training Data Distribution...")
    
    try:
        # Load original data
        df = pd.read_csv("./Dataset/phl_exoplanet_catalog.csv")
        
        if 'P_HABITABLE' in df.columns:
            habitable_counts = df['P_HABITABLE'].value_counts().sort_index()
            total = len(df)
            
            print(f"Total planets in dataset: {total}")
            print(f"Habitability class distribution:")
            
            class_names = {
                0: 'Non-Habitable',
                1: 'Potentially Habitable',
                2: 'Highly Habitable'
            }
            
            for class_id, count in habitable_counts.items():
                percentage = (count / total) * 100
                class_name = class_names.get(class_id, f'Class {class_id}')
                print(f"  {class_name}: {count} planets ({percentage:.1f}%)")
            
            # Check Earth-like planets
            print(f"\n🌍 Looking for Earth-like planets (radius ~1, mass ~1):")
            earth_like = df[
                (df['P_RADIUS'].between(0.9, 1.1)) & 
                (df['P_MASS'].between(0.9, 1.1))
            ]
            
            print(f"  Found {len(earth_like)} Earth-like planets")
            if len(earth_like) > 0:
                print(f"  Their habitability classes:")
                for class_id in [0, 1, 2]:
                    count = len(earth_like[earth_like['P_HABITABLE'] == class_id])
                    if count > 0:
                        class_name = class_names.get(class_id, f'Class {class_id}')
                        print(f"    {class_name}: {count}")
        
        else:
            print("❌ P_HABITABLE column not found in dataset")
            
    except Exception as e:
        print(f"❌ Error loading training data: {e}")

if __name__ == "__main__":
    debug_model()
    check_training_data()