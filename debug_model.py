import joblib
import pandas as pd
import json
import config

try:
    print("Loading simplified model...")
    model_path = config.MODELS_DIR / 'production' / 'simplified_8feature_model.pkl'
    model = joblib.load(model_path)
    print("Model loaded successfully")
    
    print("Loading feature names...")
    feature_path = config.MODELS_DIR / 'production' / 'simplified_features.json'
    with open(feature_path, 'r') as f:
        features = json.load(f)['features']
    print(f"Features loaded: {features}")
    
    print("Testing Proxima b prediction...")
    test_input = {
        "P_MASS_EST": 1.3,
        "P_RADIUS_EST": 1.1,
        "P_TEMP_EQUIL": 234,
        "P_PERIOD": 11.2,
        "P_FLUX": 0.65,
        "S_MASS": 0.12,
        "S_RADIUS": 0.14,
        "S_TEMPERATURE": 3050  # Note: Model expects S_TEMPERATURE
    }
    
    df = pd.DataFrame([test_input])[features]
    pred = model.predict(df)
    proba = model.predict_proba(df)
    
    print(f"Prediction: {pred[0]}")
    print(f"Probabilities: {proba[0]}")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
