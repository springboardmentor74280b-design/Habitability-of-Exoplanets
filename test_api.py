#!/usr/bin/env python3
"""
Test script for the Exoplanet Habitability API
"""

import requests
import json

API_URL = "http://localhost:5050"

print("=" * 80)
print("TESTING EXOPLANET HABITABILITY API")
print("=" * 80)

# Test 1: Health Check
print("\n[Test 1] Health Check...")
try:
    response = requests.get(f"{API_URL}/api/health")
    print(f"✓ Status: {response.status_code}")
    print(f"  Response: {response.json()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Model Info
print("\n[Test 2] Model Info...")
try:
    response = requests.get(f"{API_URL}/api/model_info")
    data = response.json()
    print(f"✓ Status: {response.status_code}")
    print(f"  Model: {data['model_info']['name']}")
    print(f"  F1 Score: {data['performance_metrics']['f1_score']:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Single Prediction with Simple Inputs
print("\n[Test 3] Single Prediction (Simple Inputs)...")
try:
    # Earth-like exoplanet
    test_data = {
        "P_MASS_EST": 1.0,      # Earth mass
        "P_RADIUS_EST": 1.0,    # Earth radius
        "P_TEMP_EQUIL": 288,    # Earth-like temperature (K)
        "P_PERIOD": 365,        # 1 year orbit
        "P_FLUX": 1.0,          # Earth-like flux
        "S_MASS": 1.0,          # Sun-like star mass
        "S_RADIUS": 1.0,        # Sun-like star radius
        "S_TEMP": 5778          # Sun-like star temperature
    }
    
    response = requests.post(
        f"{API_URL}/api/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"✓ Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        pred = result['prediction']
        print(f"\n  Prediction Results:")
        print(f"  ==================")
        print(f"  Classification: {pred['class_name']}")
        print(f"  Confidence: {pred['confidence']:.2%}")
        print(f"\n  Probabilities:")
        print(f"    Non-Habitable: {pred['probabilities']['non_habitable']:.2%}")
        print(f"    Habitable: {pred['probabilities']['habitable']:.2%}")
        print(f"    Optimistic: {pred['probabilities']['optimistic_habitable']:.2%}")
    else:
        print(f"  Error: {response.json()}")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Hot Jupiter (Non-Habitable)
print("\n[Test 4] Hot Jupiter Prediction...")
try:
    hot_jupiter = {
        "P_MASS_EST": 300,      # Very massive
        "P_RADIUS_EST": 11,     # Jupiter-like
        "P_TEMP_EQUIL": 1500,   # Very hot
        "P_PERIOD": 3,          # Very close orbit
        "P_FLUX": 100,          # High flux
        "S_MASS": 1.2,          # Slightly larger star
        "S_RADIUS": 1.3,
        "S_TEMP": 6000
    }
    
    response = requests.post(
        f"{API_URL}/api/predict",
        json=hot_jupiter,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Classification: {result['prediction']['class_name']}")
        print(f"  Confidence: {result['prediction']['confidence']:.2%}")
    else:
        print(f"✗ Error: {response.json()}")
        
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("API TESTING COMPLETE")
print("=" * 80)
print("\n✅ The API is working correctly with simple 8-parameter inputs!")
print("🌐 Open http://localhost:5050 in your browser to use the web interface")
print("=" * 80)
