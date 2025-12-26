import pandas as pd

def drop_useless_columns(df, cols_to_drop):
    return df.drop(columns=cols_to_drop, errors="ignore")

def impute_missing(df, num_cols, cat_cols):
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df
