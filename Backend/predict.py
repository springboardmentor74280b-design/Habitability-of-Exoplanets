"""
FastAPI APIRouter for exoplanet habitability prediction.

This module loads a trained sklearn pipeline (scaling, encoding, optional PCA, 
and class-weighted SVM) and provides prediction endpoints.
"""
import json
import logging
import os
from functools import lru_cache
from typing import Annotated, List

import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from schemas import ExoplanetFeaturesRequest, HabitabilityResponse


logger = logging.getLogger(__name__)

# Create APIRouter with prefix "/predict"
router = APIRouter(prefix="/predict", tags=["prediction"])

# Class label mapping: 0 = Non-Habitable, 1 = Habitable, 2 = Likely Habitable
CLASS_LABELS = {
    0: "Non-Habitable",
    1: "Habitable",
    2: "Likely Habitable",
}


class NamedFeatures(BaseModel):
    """Pydantic schema for named exoplanet features."""
    planet_radius: float = Field(..., description="Planet radius in Earth radii")
    planet_mass: float = Field(..., description="Planet mass in Earth masses")
    orbital_period: float = Field(..., description="Orbital period in days")
    stellar_temperature: float = Field(..., description="Stellar temperature in Kelvin")
    stellar_luminosity: float = Field(..., description="Stellar luminosity in solar luminosities")
    planet_density: float = Field(..., description="Planet density in g/cm³")
    semi_major_axis: float = Field(..., description="Semi-major axis in AU")


class ExoplanetFeaturesRequest(BaseModel):
    """Request schema accepting named features object."""
    features: NamedFeatures





class HabitabilityResponse(BaseModel):
    """Response schema for habitability prediction."""
    habitability_class: str = Field(..., description="Predicted habitability class")
    confidence: float = Field(..., description="Prediction confidence (max probability)")
    model: str = Field(default="Class-weighted SVM (3-class)", description="Model identifier")


def _get_model_path() -> str:
    """Resolve model path, checking Backend/model first, then saved_models."""
    backend_dir = os.path.dirname(__file__)
    
    # Try Backend/model first
    model_path = os.path.join(backend_dir, "model", "exoplanet_habitability_pipeline.pkl")
    if os.path.exists(model_path):
        return model_path
    
    # Fall back to saved_models
    project_root = os.path.abspath(os.path.join(backend_dir, ".."))
    return os.path.join(project_root, "saved_models", "exoplanet_habitability_pipeline.pkl")


def _get_feature_order_path() -> str:
    """Resolve feature order path."""
    backend_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(backend_dir, ".."))
    return os.path.join(project_root, "saved_models", "feature_order.json")


@lru_cache(maxsize=1)
def _load_pipeline():
    """
    Load the trained ML pipeline once on startup.
    
    Uses @lru_cache to ensure the model is loaded only once per process
    and reused across all API requests.
    """
    model_path = _get_model_path()
    feature_order_path = _get_feature_order_path()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if not os.path.exists(feature_order_path):
        raise FileNotFoundError(f"Feature order file not found at {feature_order_path}")
    
    logger.info(f"Loading pipeline from {model_path}")
    pipeline = joblib.load(model_path)
    
    with open(feature_order_path, "r") as f:
        feature_order = json.load(f)
    
    logger.info(f"Pipeline loaded successfully. Expected {len(feature_order)} features.")
    return pipeline, feature_order


def get_pipeline():
    """FastAPI dependency to inject the loaded pipeline."""
    return _load_pipeline()


def _map_named_features_to_array(named_features: dict, feature_order: List[str]) -> List[float]:
    """
    Map named features to the full feature array expected by the model.
    
    For features not provided, uses 0.0 as default.
    """
    # Mapping from user-friendly names to model feature names
    feature_mapping = {
        'planet_radius': 'P_RADIUS',
        'planet_mass': 'P_MASS',
        'orbital_period': 'P_PERIOD',
        'stellar_temperature': 'S_TEMPERATURE',
        'stellar_luminosity': 'S_LUMINOSITY',
        'planet_density': 'P_DENSITY',
        'semi_major_axis': 'P_SEMI_MAJOR_AXIS',
    }
    
    # Create a dict with model feature names
    model_features = {}
    for user_name, model_name in feature_mapping.items():
        if user_name in named_features:
            model_features[model_name] = named_features[user_name]
    
    # Build feature array in the correct order
    feature_array = []
    for feature_name in feature_order:
        feature_array.append(model_features.get(feature_name, 0.0))
    
    return feature_array


@router.post("", response_model=HabitabilityResponse)
def predict_habitability(
    request: ExoplanetFeaturesRequest,
    pipeline_data: Annotated[tuple, Depends(get_pipeline)],
) -> HabitabilityResponse:
    pipeline, feature_order = pipeline_data

    try:
        features_dict = request.features.model_dump()

        feature_array = _map_named_features_to_array(
            features_dict, feature_order
        )

        features_array = np.array(
            feature_array, dtype=np.float64
        ).reshape(1, -1)

        prediction = pipeline.predict(features_array)[0]
        probabilities = pipeline.predict_proba(features_array)[0]

        predicted_class = int(prediction)
        habitability_class = CLASS_LABELS.get(
            predicted_class, f"Unknown class {predicted_class}"
        )

        confidence = round(float(np.max(probabilities)), 3)

        return HabitabilityResponse(
            habitability_class=habitability_class,
            confidence=confidence,
            model="Class-weighted SVM (3-class)",
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prediction: {str(e)}"
        )

print(ExoplanetFeaturesRequest)
