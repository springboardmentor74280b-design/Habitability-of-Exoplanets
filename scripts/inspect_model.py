#!/usr/bin/env python3
"""Inspect the saved pipeline to discover expected input features and encoders.

Run: python scripts/inspect_model.py
"""
import os
import joblib
import traceback
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "notebook", "api", "model.pkl")

print(f"Project root: {PROJECT_ROOT}")
print(f"Model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("ERROR: model.pkl not found at expected location.")
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model object: {type(model)}")
except Exception:
    print("Failed to load model.pkl:")
    traceback.print_exc()
    raise


def inspect(obj, name="model", _depth=0):
    indent = "  " * _depth
    print(f"{indent}{name}: {type(obj)}")
    # common attrs
    for attr in ("n_features_in_", "feature_names_in_"):
        if hasattr(obj, attr):
            print(f"{indent}  {attr}: {getattr(obj, attr)}")
    # ColumnTransformer
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
    except Exception:
        ColumnTransformer = None
        OneHotEncoder = None
        Pipeline = None

    # If it's a pipeline, inspect steps
    if hasattr(obj, "named_steps"):
        for step_name, step in obj.named_steps.items():
            inspect(step, name=f"{name}.{step_name}", _depth=_depth + 1)

    # If it's a ColumnTransformer, list transformers
    if ColumnTransformer and isinstance(obj, ColumnTransformer):
        for t_name, transformer, cols in obj.transformers_:
            print(f"{indent}  transformer: {t_name}, cols={cols}, type={type(transformer)}")
            if OneHotEncoder and hasattr(transformer, "categories_"):
                print(f"{indent}    OneHotEncoder categories: {transformer.categories_}")
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(cols)
                    print(f"{indent}    feature names (get_feature_names_out): {names}")
                except Exception:
                    pass

    # If it's an OneHotEncoder stand-alone
    if OneHotEncoder and isinstance(obj, OneHotEncoder):
        print(f"{indent}  categories_: {obj.categories_}")
        try:
            print(f"{indent}  get_feature_names_out: {obj.get_feature_names_out()}")
        except Exception:
            pass

    # Print sample transform if possible
    if hasattr(obj, "transform") and _depth == 0:
        sample = np.zeros((1, 17))
        try:
            out = obj.transform(sample)
            print(f"{indent} transform(sample) shape: {getattr(out, 'shape', 'unknown')}")
        except Exception as e:
            print(f"{indent} transform(sample) failed: {e}")


inspect(model, "model")
print("\nDone.")
