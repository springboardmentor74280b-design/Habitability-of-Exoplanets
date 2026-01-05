from database import db
from datetime import datetime

class Exoplanet(db.Model):
    __tablename__ = "exoplanets"

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=True)

    radius = db.Column(db.Float, nullable=False)
    mass = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    orbital_period = db.Column(db.Float, nullable=False)

    habitability_score = db.Column(db.Float, nullable=True)
    is_habitable = db.Column(db.Boolean, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
