# test_integration.py
import requests
import json

def test_api_integration():
    """Test the complete API integration"""
    print("🔍 Testing API Integration...")
    
    base_url = "http://localhost:5000"
    
    # Test 1: Health check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check: {health_data}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
    
    # Test 2: Sample data
    print("\n2️⃣ Testing Sample Data...")
    try:
        response = requests.get(f"{base_url}/api/sample_data")
        if response.status_code == 200:
            sample_data = response.json()
            print(f"✅ Sample Data: {len(sample_data['samples'])} samples loaded")
            for sample in sample_data['samples']:
                print(f"   • {sample['name']}: {sample['description']}")
        else:
            print(f"❌ Sample Data Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Sample Data Error: {e}")
    
    # Test 3: Prediction with Earth-like parameters
    print("\n3️⃣ Testing Prediction (Earth-like)...")
    earth_data = {
        'P_RADIUS': 1.0,
        'P_MASS': 1.0,
        'P_GRAVITY': 1.0,
        'P_TEMP_EQUIL': 288.0,
        'P_ORBPER': 365.25,
        'P_DENSITY': 5.51,
        'S_LUM': 1.0
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json=earth_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction Successful!")
            print(f"   Prediction: {prediction['prediction_label']}")
            print(f"   Probability: {prediction['probability']}%")
            print(f"   Confidence: {prediction['confidence']}")
            print(f"   Model Used: {prediction['model_used']}")
            print(f"   Earth Similarity: {prediction['earth_similarity']}%")
            
            # Check probabilities
            probs = prediction['probabilities']
            print(f"   Probabilities:")
            print(f"     • Non-Habitable: {probs['Non_Habitable']}%")
            print(f"     • Potentially Habitable: {probs['Potentially_Habitable']}%")
            print(f"     • Highly Habitable: {probs['Highly_Habitable']}%")
            
            return True
        else:
            print(f"❌ Prediction Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return False
    
    # Test 4: Features info
    print("\n4️⃣ Testing Features Info...")
    try:
        response = requests.get(f"{base_url}/api/features")
        if response.status_code == 200:
            features_data = response.json()
            print(f"✅ Features Info:")
            print(f"   Total Features: {features_data['total_features']}")
            print(f"   Planetary Features: {features_data['feature_categories']['planetary']}")
            print(f"   Stellar Features: {features_data['feature_categories']['stellar']}")
            print(f"   Other Features: {features_data['feature_categories']['other']}")
        else:
            print(f"❌ Features Info Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Features Info Error: {e}")

if __name__ == "__main__":
    test_api_integration()