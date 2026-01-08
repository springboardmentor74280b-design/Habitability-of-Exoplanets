from web_app import create_app, db
from web_app.models import Exoplanet

app = create_app()
with app.app_context():
    # Check Kepler-410 A b (from user screenshot)
    p = Exoplanet.query.filter(Exoplanet.P_NAME.ilike('%Kepler-410 A b%')).first()
    if p:
        print(f"Name: {p.P_NAME}, Surface Temp: {p.P_TEMP_SURF}")
    else:
        print("Planet not found")
        
    # Check Earth
    p2 = Exoplanet.query.filter(Exoplanet.P_NAME.ilike('%Earth%')).first()
    if p2:
        print(f"Name: {p2.P_NAME}, Surface Temp: {p2.P_TEMP_SURF}")
