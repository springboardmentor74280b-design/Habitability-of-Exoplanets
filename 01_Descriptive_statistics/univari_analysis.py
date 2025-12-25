import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS PROJECT DOC\exoplanet_habitable_pres_dataset.csv")
df.describe()

#Distribution Plots
#Use histogram or KDE for numerical variables
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = df.select_dtypes(include=['int64','float64']).columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
#    plt.show()

#boxplot for finding ouutliers
for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


