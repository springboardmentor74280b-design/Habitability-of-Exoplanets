import pandas as pd
from web_app import create_app, db
from web_app.models import Exoplanet
import os

app = create_app()

def seed():
    data_path = "Dataset/hwc.csv" 
    # Use absolute or relative path logic
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Rename columns to match Model if needed, or map them
    # We used P_NAME, P_MASS etc in the model.
    # Check what hwc.csv has.
    # Assuming hwc.csv has standard PHL columns (P_NAME, P_MASS, etc.)
    
    print("Seeding database...")
    with app.app_context():
        db.create_all()
        
        # Check if empty
        if Exoplanet.query.first():
            print("Database already contains data. Dropping and re-seeding...")
            Exoplanet.query.delete()
            db.session.commit()

        # Bulk Insert
        # We need to map row values to Exoplanet object
        # This can be slow if we iterate.
        # fast check of columns
        records = []
        for _, row in df.iterrows():
            # Handle NaN
            row = row.fillna({
                'P_NAME': 'Unknown',
                'P_MASS': 0,
                'P_RADIUS': 0
                # ... add safe defaults dict
            })
            
            # Map Row to Model (Be careful with missing columns in CSV)
            # Use .get() method on row (which is a Series)
            
            exoplanet = Exoplanet(
                P_NAME = row.get('P_NAME', f"Planet {_}"),
                P_STATUS = str(row.get('P_STATUS', '')),
                P_ZONE_CLASS = str(row.get('P_ZONE_CLASS', '')),
                P_MASS = float(row.get('P_MASS', 0)),
                P_RADIUS = float(row.get('P_RADIUS', 0)),
                P_YEAR = float(row.get('P_YEAR', 0)),
                P_PERIOD = float(row.get('P_PERIOD', 0)),
                P_SEMI_MAJOR_AXIS = float(row.get('P_SEMI_MAJOR_AXIS', 0)),
                P_ECCENTRICITY = float(row.get('P_ECCENTRICITY', 0)),
                P_INCLINATION = float(row.get('P_INCLINATION', 0)),
                P_FLUX = float(row.get('P_FLUX', 0)),
                P_TEMP_SURF = float(row.get('P_TEMP_SURF', 0)),
                P_GRAVITY = float(row.get('P_GRAVITY', 0)),
                P_DENSITY = float(row.get('P_DENSITY', 0)),
                S_NAME = str(row.get('S_NAME', '')),
                S_TYPE = str(row.get('S_TYPE', '')),
                S_MASS = float(row.get('S_MASS', 0)),
                S_RADIUS = float(row.get('S_RADIUS', 0)),
                S_MAG = float(row.get('S_MAG', 0)),
                S_DISTANCE = float(row.get('S_DISTANCE', 0)),
                S_METALLICITY = float(row.get('S_METALLICITY', 0)),
                S_AGE = float(row.get('S_AGE', 0)),
                S_TEMPERATURE = float(row.get('S_TEMPERATURE', 0)),
                S_CONSTELLATION = str(row.get('S_CONSTELLATION', '')),
                P_HABITABLE = int(row.get('P_HABITABLE', 0))
            )
            records.append(exoplanet)
        
        # Batch insert
        db.session.add_all(records)
        db.session.commit()
        print(f"Seeded {len(records)} exoplanets.")

if __name__ == "__main__":
    seed()
