from web_app.services import predict_single
import pandas as pd

def check_kepler_1653_b():
    # Manual Input for Kepler-1653 b (from screenshot/CSV values)
    data = {
        'P_MASS': 5.35,
        'P_RADIUS': 2.17,
        'P_FLUX': 1.04, # Approx
        'P_PERIOD': 140.25,
        'P_SEMI_MAJOR_AXIS': 0.4706,
        'P_GRAVITY': 1.13, # Manually provided in form now? Or derived. Let's provide it.
        'S_TEMPERATURE': 4807,
        'S_MASS': 0.72,
        'S_RADIUS': 0.69
    }
    
    print("\nPredicting for Kepler-1653 b (Manual Input)...")
    result = predict_single(data)
    print(f"Result: {result}")
    
    # Check Kepler-452 b (Famous Habitable)
    data2 = {
        'P_MASS': 5.0, # Approx
        'P_RADIUS': 1.63,
        'P_FLUX': 1.1,
        'P_PERIOD': 384.8,
        'P_SEMI_MAJOR_AXIS': 1.046,
        'S_TEMPERATURE': 5757,
        'S_RADIUS': 1.11
    }
    print("\nPredicting for Kepler-452 b...")
    result2 = predict_single(data2)
    print(f"Result: {result2}")

if __name__ == "__main__":
    check_kepler_1653_b()
