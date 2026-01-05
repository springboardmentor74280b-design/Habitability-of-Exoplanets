import os
import joblib

# Build the correct path to the model (artifacts folder is one level up from backend)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/final_model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)

# Try to get feature names, whether it's a pipeline or a raw model
try:
    # If the model is an sklearn pipeline
    feature_names = model.named_steps['model'].feature_names_in_
except (AttributeError, KeyError):
    # If it's just a plain model
    feature_names = model.feature_names_in_

print("Features the model expects:")
for f in feature_names:
    print(f)
