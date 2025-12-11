import pandas as pd
import numpy as np

# basic info
df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS PROJECT DOC\exoplanet_habitable_pres_dataset.csv")
df.info()

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

