import json

def map_features(input_data):
    with open("feature_order.json") as f:
        order = json.load(f)
    return [input_data[feature] for feature in order]
