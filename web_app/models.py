from . import db

class Exoplanet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Basic Info
    P_NAME = db.Column(db.String(100))
    P_STATUS = db.Column(db.String(50))
    P_ZONE_CLASS = db.Column(db.String(50))
    
    # Features (Matching Model Inputs)
    # Mass, Radius, Flux, Temp, Period, Distance, etc.
    # We store what we need for display and potentially for prediction if we want to "re-predict" known ones
    P_MASS = db.Column(db.Float)
    P_RADIUS = db.Column(db.Float)
    P_YEAR = db.Column(db.Float)
    P_PERIOD = db.Column(db.Float)
    P_SEMI_MAJOR_AXIS = db.Column(db.Float)
    P_ECCENTRICITY = db.Column(db.Float)
    P_INCLINATION = db.Column(db.Float)
    P_FLUX = db.Column(db.Float)
    P_TEMP_SURF = db.Column(db.Float)
    P_GRAVITY = db.Column(db.Float)
    P_DENSITY = db.Column(db.Float)
    
    # Star Info
    S_NAME = db.Column(db.String(100))
    S_TYPE = db.Column(db.String(50))
    S_MASS = db.Column(db.Float)
    S_RADIUS = db.Column(db.Float)
    S_MAG = db.Column(db.Float)
    S_DISTANCE = db.Column(db.Float)
    S_METALLICITY = db.Column(db.Float)
    S_AGE = db.Column(db.Float)
    S_TEMPERATURE = db.Column(db.Float)
    S_CONSTELLATION = db.Column(db.String(50))
    
    # Target
    P_HABITABLE = db.Column(db.Integer) # 0, 1, 2
    
