"""
Test script to check predictions for all test samples
"""
import requests
import json

API_URL = "http://localhost:5050/api/predict"

# Load test samples
with open('outputs/planet_presets.json', 'r') as f:
    data = json.load(f)

test_samples = data['test_samples']

print("=" * 80)
print("TESTING ALL CUSTOM TEST SAMPLES")
print("=" * 80)

results = []

for i, sample in enumerate(test_samples, 1):
    print(f"\n{i}. {sample['name']} - {sample['description']}")
    
    # Prepare data for API
    test_data = sample['data'].copy()
    
    try:
        response = requests.post(API_URL, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            
            print(f"   Prediction: {prediction['class_name']}")
            print(f"   Confidence: {prediction['confidence']*100:.2f}%")
            print(f"   Probabilities:")
            print(f"     - Non-Habitable: {prediction['probabilities']['non_habitable']*100:.1f}%")
            print(f"     - Habitable: {prediction['probabilities']['habitable']*100:.1f}%")
            
            results.append({
                'name': sample['name'],
                'description': sample['description'],
                'prediction': prediction['class_name'],
                'confidence': prediction['confidence']
            })
        else:
            print(f"   ERROR: {response.status_code} - {response.text}")
            results.append({
                'name': sample['name'],
                'description': sample['description'],
                'prediction': 'ERROR',
                'confidence': 0
            })
    except Exception as e:
        print(f"   ERROR: {str(e)}")
        results.append({
            'name': sample['name'],
            'description': sample['description'],
            'prediction': 'ERROR',
            'confidence': 0
        })

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

habitable_count = sum(1 for r in results if r['prediction'] == 'Habitable')
non_habitable_count = sum(1 for r in results if r['prediction'] == 'Non-Habitable')
error_count = sum(1 for r in results if r['prediction'] == 'ERROR')

print(f"\nTotal Test Samples: {len(results)}")
print(f"Habitable: {habitable_count}")
print(f"Non-Habitable: {non_habitable_count}")
print(f"Errors: {error_count}")

if non_habitable_count == len(results):
    print("\n⚠️  WARNING: ALL test samples are predicted as Non-Habitable!")
else:
    print(f"\n✓ Mixed predictions found")

print("\n" + "=" * 80)
