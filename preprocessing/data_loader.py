import pandas as pd

def load_data(path="Dataset/hwc.csv"):
    df = pd.read_csv(path)
    return df
