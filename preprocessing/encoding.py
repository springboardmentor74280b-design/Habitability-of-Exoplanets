import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def hybrid_encode(X_train, X_test, cat_cols, threshold=10):
    """
    Applies One-Hot Encoding to low cardinality columns and 
    Frequency Encoding to high cardinality columns.
    
    Returns X_train, X_test (as DataFrames), and lists of new column names.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    ohe_cols = []
    freq_cols = []
    
    # Identify which columns go where based on TRAIN set cardinality
    for col in cat_cols:
        if X_train[col].nunique() <= threshold:
            ohe_cols.append(col)
        else:
            freq_cols.append(col)
            
    print(f"   > One-Hot Cols (<= {threshold}): {ohe_cols}")
    print(f"   > Freq Cols (> {threshold}): {freq_cols}")
    
    # 1. Frequency Encoding (High Cardinality)
    # Map counts from Train to Test to prevent leakage and ensure consistency
    for col in freq_cols:
        freq_map = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(freq_map)
        # Use simple map; unseen categories in Test become NaN -> fill with 0
        X_test[col] = X_test[col].map(freq_map).fillna(0)
    
    # 2. One-Hot Encoding (Low Cardinality)
    if ohe_cols:
        # handle_unknown='ignore' ensures new categories in test don't crash the code
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype='int')
        
        # Fit on Train
        ohe.fit(X_train[ohe_cols])
        
        # Transform Train
        train_ohe = ohe.transform(X_train[ohe_cols])
        new_ohe_cols = ohe.get_feature_names_out(ohe_cols)
        train_ohe_df = pd.DataFrame(train_ohe, columns=new_ohe_cols, index=X_train.index)
        
        # Transform Test
        test_ohe = ohe.transform(X_test[ohe_cols])
        test_ohe_df = pd.DataFrame(test_ohe, columns=new_ohe_cols, index=X_test.index)
        
        # Drop original cat columns and concat OHE columns
        X_train = pd.concat([X_train.drop(columns=ohe_cols), train_ohe_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=ohe_cols), test_ohe_df], axis=1)
    
    return X_train, X_test, freq_cols
