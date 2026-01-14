import pandas as pd
import plotly.express as px
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(ARTIFACTS_DIR, "ranked_exoplanets.csv")

df = pd.read_csv(DATA_PATH)

fig1 = px.scatter(
    df,
    x="pl_rade",
    y="habitability_score",
    color="habitability_score",
    title="Planet Radius vs Habitability Score",
    labels={
        "pl_rade": "Planet Radius (Earth Radii)",
        "habitability_score": "Habitability Score"
    }
)

fig2 = px.scatter(
    df,
    x="pl_eqt",
    y="habitability_score",
    color="habitability_score",
    title="Temperature vs Habitability Score",
    labels={
        "pl_eqt": "Equilibrium Temperature (K)",
        "habitability_score": "Habitability Score"
    }
)

fig1.write_html(os.path.join(REPORTS_DIR, "radius_vs_score.html"))
fig2.write_html(os.path.join(REPORTS_DIR, "temperature_vs_score.html"))

print("✅ Interactive Plotly charts saved")
