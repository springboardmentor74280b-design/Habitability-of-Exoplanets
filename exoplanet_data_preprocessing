import pandas as pd

df = pd.read_csv("exoplanet.csv")
df = df.dropna()

features = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_eqt']
df = df[features]

df.to_csv("clean_exoplanet.csv", index=False)
