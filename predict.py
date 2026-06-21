import joblib
import pandas as pd

model = joblib.load("model/exoplanet_model.pkl")

def predict_habitability(
planet_radius,
planet_mass,
orbital_period,
star_temperature,
star_luminosity
):

```
data = pd.DataFrame([{
    "Planet_Radius": planet_radius,
    "Planet_Mass": planet_mass,
    "Orbital_Period": orbital_period,
    "Star_Temperature": star_temperature,
    "Star_Luminosity": star_luminosity
}])

prediction = model.predict(data)[0]

probability = model.predict_proba(data)[0][1]

return prediction, round(probability * 100, 2)
```

if **name** == "**main**":

```
result, score = predict_habitability(
    1.1,
    1.2,
    365,
    5778,
    1.0
)

print("Prediction:", result)
print("Habitability Score:", score, "%")
```
