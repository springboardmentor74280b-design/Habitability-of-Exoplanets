#!/usr/bin/env python3
"""Check that notebook/api/model.pkl can be loaded and used for a sample prediction.

Usage:
  python scripts/check_model.py

This script prints diagnostic information and any traceback that occurs during loading or prediction.
"""
import os
import traceback
import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "notebook", "api", "model.pkl")

print(f"Project root: {PROJECT_ROOT}")
print(f"Model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("ERROR: model.pkl not found at expected location.")
    print("Make sure the trained model is saved to notebook/api/model.pkl")
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model object: {type(model)}")
except Exception:
    print("Failed to load model.pkl:")
    traceback.print_exc()
    raise

# Try a sample predict_proba if available
sample = np.array([[0.5, 5.5, 280, 1.1, 1.2, 3500, 0.02, 1, 0, 0]])
print(f"Sample input shape: {sample.shape}")

try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)
        print("predict_proba output:", probs)
    elif hasattr(model, "predict"):
        preds = model.predict(sample)
        print("predict output:", preds)
    else:
        print("Model has neither predict_proba nor predict method; inspect the object.")
except Exception:
    print("Error during model inference:")
    traceback.print_exc()
    raise

print("Model load and inference check completed successfully.")
