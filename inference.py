import joblib
import numpy as np

# Load model + scaler + feature order
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")  # should contain the 7 feature names

def predict_habitability_score(input_features: dict):
    # Order input according to features.pkl
    x = np.array([[input_features[f] for f in features]], dtype=float)

    # Scale inputs
    x_scaled = scaler.transform(x)

    # Predict score (0–1)
    score = model.predict(x_scaled)[0]
    return score
