"""
Utility module for mapping user-friendly feature names to model feature vector.
"""
import json
import os
from typing import Dict, List, Optional
import numpy as np


# Mapping from user-friendly names to model feature names
FEATURE_MAPPING = {
    # Planetary features
    "planet_radius": "P_RADIUS",
    "planet_mass": "P_MASS",
    "planet_density": "P_DENSITY",
    "surface_temperature": "P_TEMP_SURF",
    "orbital_period": "P_PERIOD",
    "distance_from_star": "P_SEMI_MAJOR_AXIS",
    "eccentricity": "P_ECCENTRICITY",
    "equilibrium_temperature": "P_TEMP_EQUIL",
    "gravity": "P_GRAVITY",
    "escape_velocity": "P_ESCAPE",
    
    # Stellar features
    "host_star_temperature": "S_TEMPERATURE",
    "host_star_luminosity": "S_LUMINOSITY",
    "host_star_metallicity": "S_METALLICITY",
    "host_star_mass": "S_MASS",
    "host_star_radius": "S_RADIUS",
    "host_star_age": "S_AGE",
    "host_star_distance": "S_DISTANCE",
}


def load_feature_order() -> List[str]:
    """Load the feature order from saved_models/feature_order.json"""
    feature_order_path = os.path.join(
        os.path.dirname(__file__), "..", "saved_models", "feature_order.json"
    )
    with open(feature_order_path, "r") as f:
        return json.load(f)


def map_features_to_vector(
    planetary_features,
    stellar_features,
    feature_order: Optional[List[str]] = None,
    default_value: float = 0.0
) -> List[float]:
    """
    Map user-friendly features to the full feature vector required by the model.
    
    Args:
        planetary_features: PlanetaryFeatures Pydantic model or dict with planetary feature values
        stellar_features: StellarFeatures Pydantic model or dict with stellar feature values
        feature_order: List of feature names in the order expected by the model
        default_value: Value to use for features not provided (default: 0.0)
    
    Returns:
        List of feature values in the order expected by the model
    """
    if feature_order is None:
        feature_order = load_feature_order()
    
    # Convert Pydantic models to dicts if needed
    if hasattr(planetary_features, 'model_dump'):
        # Pydantic v2
        planetary_dict = planetary_features.model_dump()
    elif hasattr(planetary_features, 'dict'):
        # Pydantic v1
        planetary_dict = planetary_features.dict()
    else:
        planetary_dict = planetary_features
    
    if hasattr(stellar_features, 'model_dump'):
        # Pydantic v2
        stellar_dict = stellar_features.model_dump()
    elif hasattr(stellar_features, 'dict'):
        # Pydantic v1
        stellar_dict = stellar_features.dict()
    else:
        stellar_dict = stellar_features
    
    # Create a dictionary mapping model feature names to values
    feature_dict = {}
    
    # Map planetary features
    planetary_mappings = {
        "planet_radius": "P_RADIUS",
        "planet_mass": "P_MASS",
        "planet_density": "P_DENSITY",
        "surface_temperature": "P_TEMP_SURF",
        "orbital_period": "P_PERIOD",
        "distance_from_star": "P_SEMI_MAJOR_AXIS",
        "eccentricity": "P_ECCENTRICITY",
        "equilibrium_temperature": "P_TEMP_EQUIL",
        "gravity": "P_GRAVITY",
        "escape_velocity": "P_ESCAPE",
    }
    
    for user_name, model_name in planetary_mappings.items():
        value = planetary_dict.get(user_name)
        if value is not None:
            feature_dict[model_name] = value
    
    # Map stellar features
    stellar_mappings = {
        "host_star_temperature": "S_TEMPERATURE",
        "host_star_luminosity": "S_LUMINOSITY",
        "host_star_metallicity": "S_METALLICITY",
        "host_star_mass": "S_MASS",
        "host_star_radius": "S_RADIUS",
        "host_star_age": "S_AGE",
        "host_star_distance": "S_DISTANCE",
    }
    
    for user_name, model_name in stellar_mappings.items():
        value = stellar_dict.get(user_name)
        if value is not None:
            feature_dict[model_name] = value
    
    # Build the feature vector in the correct order
    feature_vector = []
    for feature_name in feature_order:
        if feature_name in feature_dict:
            feature_vector.append(feature_dict[feature_name])
        else:
            feature_vector.append(default_value)
    
    return feature_vector


def get_feature_mapping_info() -> Dict:
    """Get information about feature mappings for API documentation"""
    return {
        "mapped_features": list(FEATURE_MAPPING.keys()),
        "total_features_required": len(load_feature_order()),
        "note": "Features not provided will be set to 0.0. For accurate predictions, "
                "provide all available features or use the direct feature array endpoint."
    }
