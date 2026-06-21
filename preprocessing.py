import pandas as pd

def create_features(df):

```
df["Habitability_Index"] = (
    (1 / (abs(df["Planet_Radius"] - 1) + 1))
    +
    (1 / (abs(df["Planet_Mass"] - 1) + 1))
) / 2

df["Stellar_Compatibility_Index"] = (
    (1 / (abs(df["Star_Temperature"] - 5778) / 5778 + 1))
    +
    (1 / (abs(df["Star_Luminosity"] - 1) + 1))
) / 2

return df
```

if **name** == "**main**":

```
df = pd.read_csv("data/exoplanet_dataset.csv")

df = create_features(df)

df.to_csv(
    "data/processed_data.csv",
    index=False
)

print("Feature Engineering Completed")
```
