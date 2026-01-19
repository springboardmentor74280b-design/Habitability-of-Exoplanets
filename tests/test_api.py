"""Pytest tests for the API endpoints.

Note: The API must be running locally (http://127.0.0.1:5000) for these tests to pass.
Run: pytest -q
"""

import requests
import pytest

BASE = "http://127.0.0.1:5000"

SAMPLE = {
    "HSI": 0.5,
    "planet_density": 5.5,
    "pl_eqt": 280,
    "pl_rade": 1.1,
    "pl_bmasse": 1.2,
    "st_teff": 3500,
    "star_luminosity": 0.02,
    "star_type_M": 1,
    "star_type_K": 0,
    "star_type_G": 0,
}


def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_info():
    r = requests.get(f"{BASE}/info", timeout=5)
    assert r.status_code == 200
    j = r.json()
    assert "final_features" in j


def test_predict_ok():
    r = requests.post(f"{BASE}/predict", json=SAMPLE, timeout=5)
    assert r.status_code == 200
    j = r.json()
    assert "habitability_prediction" in j
    assert "habitability_probability" in j
    assert "filled_defaults" in j


def test_predict_missing_fields():
    r = requests.post(f"{BASE}/predict", json={"HSI": 0.1}, timeout=5)
    assert r.status_code == 400
    j = r.json()
    assert "missing" in j or "error" in j
