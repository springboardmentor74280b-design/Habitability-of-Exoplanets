#BIVARIATE ANALYSIS - Used to understand relationships between features, and 
# between features & target (e.g., habitable vs non-habitable).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS PROJECT DOC\exoplanet_habitable_pres_dataset.csv")

#correlation - To understand which features are strongly related.
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()
#corr = df.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Scatter Plots - Shows relationship between two continuous features.
#Example: star temperature vs planet radius
sns.scatterplot(x=df['S_TEMPERATURE'], y=df['P_RADIUS'])
plt.title("Star Temperature vs Planet Radius")
plt.show()

sns.scatterplot(x=df['P_DISTANCE'], y=df['P_RADIUS'])
plt.title("Planet Distance vs Planet Radius")
plt.show()




