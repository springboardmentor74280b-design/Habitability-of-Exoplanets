import pandas as pd
import numpy as np

# basic info
df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\Habitability-of-Exoplanets\exoplanet_habitable_pres_dataset.csv")
#df.info()

# count missing per column (sorted)
missing = df.isna().sum().sort_values(ascending=False)
missing[missing > 0]
print(missing)

# fraction missing
(df.isna().mean()*100).sort_values(ascending=False)

# pairwise missingness (heatmap quick look)
import seaborn as sns, matplotlib.pyplot as plt
sns.heatmap(df.isna(), cbar=False)
plt.title("Missingness map")
plt.show()

#return only the columns where missing values exist.
df.isnull().sum()[df.isnull().sum() > 0]
#print(df)



missing_columns = df.columns[df.isnull().any()].tolist()
#print(missing_columns)
for col in missing_columns:
    print(col)

#perc % of missi 
def missing_percentage(df):
    return (df.isnull().sum() * 100 / len(df)).round(2)
print(missing_percentage(df))




for col in ["P_MASS", "P_RADIUS"]:   
  df[col + "_was_missing"] = df[col].isna().astype(int)
  print(df)

################## preprocessing ############################################
def drop_high_missing(df, threshold=40):
    missing_percent = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    
    print("Dropping columns:", cols_to_drop)
    
    df = df.drop(columns=cols_to_drop)
    return df

df = drop_high_missing(df, threshold=40)

#for numerical col fill with mean and median 
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
    print(num_cols)

#for catrgorical col fill with mode
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    print(cat_cols)