import streamlit as st
from predict import predict_habitability

st.set_page_config(
page_title="Exoplanet Habitability Predictor",
layout="wide"
)

st.title("🌍 Exoplanet Habitability Prediction System")

st.write(
"Predict whether an exoplanet could support life based on planetary and stellar characteristics."
)

col1, col2 = st.columns(2)

with col1:

```
planet_radius = st.number_input(
    "Planet Radius (Earth Radius)",
    min_value=0.1,
    value=1.0
)

planet_mass = st.number_input(
    "Planet Mass (Earth Mass)",
    min_value=0.1,
    value=1.0
)

orbital_period = st.number_input(
    "Orbital Period (Days)",
    min_value=1,
    value=365
)
```

with col2:

```
star_temperature = st.number_input(
    "Star Temperature (Kelvin)",
    min_value=1000,
    value=5778
)

star_luminosity = st.number_input(
    "Star Luminosity",
    min_value=0.1,
    value=1.0
)
```

if st.button("Predict Habitability"):

```
prediction, score = predict_habitability(
    planet_radius,
    planet_mass,
    orbital_period,
    star_temperature,
    star_luminosity
)

st.subheader("Prediction Results")

st.metric(
    "Habitability Score",
    f"{score}%"
)

if prediction == 1:
    st.success("Potentially Habitable Planet")
else:
    st.error("Not Habitable")

st.progress(score / 100)

st.subheader("Planet Summary")

st.write(f"Planet Radius: {planet_radius}")
st.write(f"Planet Mass: {planet_mass}")
st.write(f"Orbital Period: {orbital_period}")
st.write(f"Star Temperature: {star_temperature}")
st.write(f"Star Luminosity: {star_luminosity}")
```
