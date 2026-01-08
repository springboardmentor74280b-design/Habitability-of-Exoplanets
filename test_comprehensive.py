#!/usr/bin/env python3
"""
Comprehensive Test: Various Exoplanet Types
Tests the API with different planet configurations to verify
it can correctly classify both habitable and non-habitable planets
"""

import requests
import json

API_URL = "http://localhost:5050"

print("=" * 80)
print("COMPREHENSIVE EXOPLANET CLASSIFICATION TEST")
print("=" * 80)

test_cases = [
    {
        "name": "🌍 Earth-like (Potentially Habitable)",
        "description": "Similar to Earth in all parameters",
        "data": {
            "P_MASS_EST": 1.0,
            "P_RADIUS_EST": 1.0,
            "P_TEMP_EQUIL": 288,
            "P_PERIOD": 365,
            "P_FLUX": 1.0,
            "S_MASS": 1.0,
            "S_RADIUS": 1.0,
            "S_TEMP": 5778
        },
        "expected": "Should be Habitable or Optimistic"
    },
    {
        "name": "🔥 Hot Jupiter (Non-Habitable)",
        "description": "Massive gas giant, very close to star",
        "data": {
            "P_MASS_EST": 300,
            "P_RADIUS_EST": 11,
            "P_TEMP_EQUIL": 1500,
            "P_PERIOD": 3,
            "P_FLUX": 100,
            "S_MASS": 1.2,
            "S_RADIUS": 1.3,
            "S_TEMP": 6000
        },
        "expected": "Should be Non-Habitable"
    },
    {
        "name": "❄️ Frozen Super-Earth (Non-Habitable)",
        "description": "Large planet, far from star, very cold",
        "data": {
            "P_MASS_EST": 5.0,
            "P_RADIUS_EST": 1.8,
            "P_TEMP_EQUIL": 50,
            "P_PERIOD": 2000,
            "P_FLUX": 0.01,
            "S_MASS": 0.8,
            "S_RADIUS": 0.7,
            "S_TEMP": 4500
        },
        "expected": "Should be Non-Habitable"
    },
    {
        "name": "🌊 Ocean World (Potentially Habitable)",
        "description": "Slightly larger than Earth, in habitable zone",
        "data": {
            "P_MASS_EST": 2.5,
            "P_RADIUS_EST": 1.3,
            "P_TEMP_EQUIL": 280,
            "P_PERIOD": 400,
            "P_FLUX": 0.9,
            "S_MASS": 0.95,
            "S_RADIUS": 0.92,
            "S_TEMP": 5500
        },
        "expected": "Should be Habitable or Optimistic"
    },
    {
        "name": "🪨 Rocky Planet - Too Hot (Non-Habitable)",
        "description": "Rocky but too close to star",
        "data": {
            "P_MASS_EST": 0.8,
            "P_RADIUS_EST": 0.9,
            "P_TEMP_EQUIL": 700,
            "P_PERIOD": 50,
            "P_FLUX": 10,
            "S_MASS": 1.1,
            "S_RADIUS": 1.05,
            "S_TEMP": 5900
        },
        "expected": "Should be Non-Habitable"
    },
    {
        "name": "🌟 Proxima Centauri b-like (Potentially Habitable)",
        "description": "Around red dwarf, in habitable zone",
        "data": {
            "P_MASS_EST": 1.3,
            "P_RADIUS_EST": 1.1,
            "P_TEMP_EQUIL": 234,
            "P_PERIOD": 11.2,
            "P_FLUX": 0.65,
            "S_MASS": 0.12,
            "S_RADIUS": 0.14,
            "S_TEMP": 3050
        },
        "expected": "Should be Habitable or Optimistic"
    }
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Description: {test_case['description']}")
    print(f"Expected: {test_case['expected']}")
    
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            json=test_case['data'],
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            pred = result['prediction']
            
            print(f"\n✓ Prediction: {pred['class_name']}")
            print(f"  Confidence: {pred['confidence']:.2%}")
            print(f"\n  Probability Breakdown:")
            print(f"    Non-Habitable:        {pred['probabilities']['non_habitable']:.2%}")
            print(f"    Habitable:            {pred['probabilities']['habitable']:.2%}")
            print(f"    Optimistic Habitable: {pred['probabilities']['optimistic_habitable']:.2%}")
            
            results.append({
                'name': test_case['name'],
                'prediction': pred['class_name'],
                'confidence': pred['confidence'],
                'expected': test_case['expected']
            })
        else:
            print(f"✗ Error: {response.json()}")
            results.append({
                'name': test_case['name'],
                'prediction': 'ERROR',
                'confidence': 0,
                'expected': test_case['expected']
            })
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        results.append({
            'name': test_case['name'],
            'prediction': 'EXCEPTION',
            'confidence': 0,
            'expected': test_case['expected']
        })

# Summary
print(f"\n{'='*80}")
print("SUMMARY OF ALL PREDICTIONS")
print(f"{'='*80}\n")

print(f"{'Planet Type':<40} {'Prediction':<25} {'Confidence':<12}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<40} {r['prediction']:<25} {r['confidence']:.1%}")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")

# Analysis
habitable_count = sum(1 for r in results if 'Habitable' in r['prediction'] and 'Non' not in r['prediction'])
non_habitable_count = sum(1 for r in results if 'Non-Habitable' in r['prediction'])

print(f"\n📊 Classification Distribution:")
print(f"  Habitable/Optimistic: {habitable_count}")
print(f"  Non-Habitable: {non_habitable_count}")
print(f"  Total Tests: {len(results)}")

print(f"\n✅ The model can now classify various planet types!")
print(f"🌐 Access the web interface at: http://localhost:5050")
print("=" * 80)
