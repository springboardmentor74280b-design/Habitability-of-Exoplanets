from model_loader import load_model
from feature_mapper import map_features

model = load_model()

def predict_habitability(data):
    features = map_features(data)
    prediction = model.predict([features])[0]
    return {"Habitability": "Habitable" if prediction == 1 else "Non-Habitable"}
