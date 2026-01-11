import pickle

def load_model():
    with open("exoplanet_habitability_pipeline.pkl", "rb") as f:
        return pickle.load(f)
