import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardize(train, test, cols_to_scale=None):
    """
    Standardizes specified columns using StandardScaler under the hood.
    Returns DataFrames to preserve column names.
    """
    # If input is numpy, convert to DF (though we expect DF from main.py)
    if not isinstance(train, pd.DataFrame):
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        
    train_out = train.copy()
    test_out = test.copy()
    
    scaler = StandardScaler()
    
    if cols_to_scale is None:
        cols_to_scale = train.columns.tolist()
        
    # Fit on Train
    train_out[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
    
    # Transform Test
    test_out[cols_to_scale] = scaler.transform(test[cols_to_scale])
    
    return train_out, test_out, scaler

def normalize(train, test):
    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.transform(test)
    return train_norm, test_norm, scaler
