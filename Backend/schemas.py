from pydantic import BaseModel, Field


# =========================
# Planetary Feature Schema
# =========================
class PlanetaryFeatures(BaseModel):
    """Planetary + stellar features used by the trained ML model"""

    planet_radius: float = Field(..., description="Planet radius in Earth radii", gt=0)
    planet_mass: float = Field(..., description="Planet mass in Earth masses", gt=0)
    orbital_period: float = Field(..., description="Orbital period in days", gt=0)

    stellar_temperature: float = Field(
        ..., description="Host star effective temperature in Kelvin", gt=0
    )
    stellar_luminosity: float = Field(
        ..., description="Host star luminosity in solar units", gt=0
    )

    planet_density: float = Field(..., description="Planet density in g/cm³", gt=0)
    semi_major_axis: float = Field(
        ..., description="Distance from star (AU)", gt=0
    )


# =========================
# Prediction Request
# =========================
class PredictRequest(BaseModel):
    """Request schema expected by frontend"""
    features: PlanetaryFeatures


# =========================
# Prediction Response
# =========================
class PredictionResponse(BaseModel):
    habitability_class: str
    confidence: float
    model: str




from pydantic import BaseModel, Field

class NamedFeatures(BaseModel):
    planet_radius: float
    planet_mass: float
    orbital_period: float
    stellar_temperature: float
    stellar_luminosity: float
    planet_density: float
    semi_major_axis: float

class ExoplanetFeaturesRequest(BaseModel):
    features: NamedFeatures

class HabitabilityResponse(BaseModel):
    habitability_class: str
    confidence: float
    model: str

