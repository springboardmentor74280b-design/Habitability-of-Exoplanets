import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS PROJECT DOC\exoplanet_habitable_pres_dataset.csv")

def drop_high_missing(df, threshold=40):
    missing_percent = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    
    print("Dropping columns:", cols_to_drop)
    
    df = df.drop(columns=cols_to_drop)
    return df


#for numerical col fill with mean and median 
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
    print(num_cols)

#for categorical col fill with mode
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    print(cat_cols)

df.head()          
df.info()          
df.isnull().sum()  # Should be 0 or very small
df.isnull().sum().sort_values(ascending=False).head(10)
df.to_csv("cleaned_exoplanet_dataset.csv", index=False)




