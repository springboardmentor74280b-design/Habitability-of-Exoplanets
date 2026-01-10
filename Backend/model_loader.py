import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import joblib


logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container for the trained ML pipeline and its metadata."""

    model: object
    feature_order: List[str]


def _resolve_path(env_var: str, default_relative: str) -> str:
    """
    Resolve a file path, preferring an environment variable override.

    The default path is interpreted relative to the project root
    (one level above this Backend package).
    """
    override = os.getenv(env_var)
    if override:
        return override

    backend_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(backend_dir, ".."))
    return os.path.join(project_root, default_relative)


def get_model_path() -> str:
    """Get model path, checking Backend/model first, then saved_models."""
    # Check for environment variable override
    override = os.getenv("MODEL_PATH")
    if override:
        return override
    
    backend_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(backend_dir, ".."))
    
    # Try Backend/model first (as specified by user)
    model_path_backend = os.path.join(backend_dir, "model", "exoplanet_habitability_pipeline.pkl")
    if os.path.exists(model_path_backend):
        return model_path_backend
    
    # Fall back to saved_models (existing structure)
    return os.path.join(project_root, "saved_models", "exoplanet_habitability_pipeline.pkl")


def get_feature_order_path() -> str:
    return _resolve_path(
        "FEATURE_ORDER_PATH",
        os.path.join("saved_models", "feature_order.json"),
    )


@lru_cache(maxsize=1)
def load_model_artifacts() -> ModelArtifacts:
    """
    Load and cache the trained ML pipeline and feature order.

    This is safe to use as a FastAPI dependency; the artifacts will be
    loaded once per process and reused across requests.
    """
    model_path = get_model_path()
    feature_order_path = get_feature_order_path()

    logger.info("Loading model from %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info("Loading feature order from %s", feature_order_path)
    if not os.path.exists(feature_order_path):
        raise FileNotFoundError(
            f"Feature order file not found at {feature_order_path}"
        )

    model = joblib.load(model_path)
    with open(feature_order_path, "r") as f:
        feature_order = json.load(f)

    if not isinstance(feature_order, list):
        raise ValueError("feature_order.json must contain a list of feature names")

    logger.info(
        "Model artifacts loaded: %d features in expected order",
        len(feature_order),
    )
    return ModelArtifacts(model=model, feature_order=feature_order)


def warm_up_model() -> None:
    """
    Eagerly load the model at startup so that the first request is fast.
    """
    try:
        _ = load_model_artifacts()
    except Exception:
        # Let the exception propagate to the caller if they choose to handle it,
        # but also log here for visibility during startup.
        logger.exception("Failed to warm up model artifacts")
        raise

