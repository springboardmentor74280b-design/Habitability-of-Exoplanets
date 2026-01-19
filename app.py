# astrohab_server.py - COMPLETE ONE-FILE SOLUTION
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import traceback
import math
import json

# ========== CONFIGURATION ==========
PORT = 5000
HOST = '0.0.0.0'

# Initialize Flask
app = Flask(__name__, static_folder='.', static_url_path='')

# Allow ALL origins for development
CORS(app, supports_credentials=True)

# ========== MODEL LOADING ==========
print("\n" + "="*60)
print("🚀 LOADING ASTROHAB XGBOOST MODEL")
print("="*60)

model = None
preprocessor = None
feature_columns = []

try:
    # Load model from saved_models folder
    model_files = {
        'model': 'saved_models/xgb_habitability_model_0.pkl',
        'preprocessor': 'saved_models/preprocessor.pkl', 
        'features': 'saved_models/feature_columns.pkl'
    }
    
    # Check if files exist
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"✅ Found {name}: {path}")
        else:
            print(f"❌ Missing {name}: {path}")
    
    # Load the model
    if os.path.exists(model_files['model']):
        model = joblib.load(model_files['model'])
        print(f"✅ XGBoost Model loaded: {type(model).__name__}")
        
        if os.path.exists(model_files['preprocessor']):
            preprocessor = joblib.load(model_files['preprocessor'])
            print(f"✅ Preprocessor loaded")
            
        if os.path.exists(model_files['features']):
            feature_columns = joblib.load(model_files['features'])
            print(f"✅ Features loaded: {len(feature_columns)} columns")
            
            # Test the model
            test_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
            if preprocessor:
                test_data = preprocessor.transform(test_data)
            prediction = model.predict(test_data)
            print(f"✅ Model test: Prediction = {prediction[0]}")
        
except Exception as e:
    print(f"❌ ERROR loading models: {e}")
    traceback.print_exc()

print("="*60 + "\n")

# ========== HELPER FUNCTIONS ==========
CLASS_LABELS = {
    0: "Non-Habitable",
    1: "Potentially Habitable", 
    2: "Highly Habitable"
}

def prepare_input_for_model(data):
    """Prepare input for XGBoost model"""
    try:
        if not feature_columns:
            return None
        
        # Initialize all features to 0
        input_dict = {col: 0.0 for col in feature_columns}
        
        # Common parameter mappings
        param_mappings = {
            'radius': ['P_RADIUS', 'radius'],
            'mass': ['P_MASS', 'mass'],
            'gravity': ['P_GRAVITY', 'gravity'],
            'temp': ['P_TEMP_EQUIL', 'temp', 'temperature'],
            'period': ['P_ORBPER', 'period', 'orbital_period'],
            'density': ['P_DENSITY', 'density']
        }
        
        # Try to map each parameter
        for param, value in data.items():
            param_lower = param.lower()
            
            # Check all feature columns for matches
            for feature in feature_columns:
                feature_lower = feature.lower()
                
                # Direct match or partial match
                if (param_lower in feature_lower or 
                    feature_lower in param_lower or
                    any(map_term in feature_lower for map_term in param_mappings.get(param_lower, []))):
                    
                    try:
                        input_dict[feature] = float(value)
                        break
                    except:
                        continue
        
        # Create DataFrame
        df = pd.DataFrame([input_dict])
        df = df[feature_columns]
        
        return df
        
    except Exception as e:
        print(f"Input preparation error: {e}")
        return None

# ========== ROUTES ==========
@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    try:
        # Try different possible locations
        paths_to_try = [
            'index.html',
            'frontend/index.html',
            './frontend/index.html'
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        # If not found, return simple page
        return """
        <html>
            <head><title>AstroHab</title></head>
            <body>
                <h1>AstroHab - Exoplanet Habitability Predictor</h1>
                <p>Place your index.html in the project folder.</p>
                <p>API is running at <a href="/api/health">/api/health</a></p>
            </body>
        </html>
        """
    except Exception as e:
        return f"Error loading page: {str(e)}"

@app.route('/<path:filename>')
def serve_file(filename):
    """Serve static files"""
    try:
        return send_from_directory('.', filename)
    except:
        return "File not found", 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'model_type': 'XGBoost',
        'features': len(feature_columns) if feature_columns else 0,
        'server': 'Flask',
        'port': PORT
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not model:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        print(f"📥 Prediction request")
        
        # Get parameters with defaults
        params = {
            'radius': float(data.get('radius', data.get('P_RADIUS', 1.0))),
            'mass': float(data.get('mass', data.get('P_MASS', 1.0))),
            'gravity': float(data.get('gravity', data.get('P_GRAVITY', 1.0))),
            'temp': float(data.get('temp', data.get('temperature', data.get('P_TEMP_EQUIL', 288.0)))),
            'period': float(data.get('period', data.get('orbital_period', data.get('P_ORBPER', 365.0)))),
            'density': float(data.get('density', data.get('P_DENSITY', 5.51)))
        }
        
        print(f"📊 Parameters: {params}")
        
        # Prepare input
        input_df = prepare_input_for_model(params)
        if input_df is None:
            return jsonify({'error': 'Input preparation failed'}), 400
        
        # Make prediction
        if preprocessor:
            features = preprocessor.transform(input_df)
        else:
            features = input_df.values
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        print(f"✅ Prediction: {prediction}, Probs: {probabilities}")
        
        # Format results
        prediction_label = CLASS_LABELS.get(int(prediction), "Unknown")
        probability_score = float(probabilities[int(prediction)] * 100)
        
        formatted_probs = {
            'Non_Habitable': round(float(probabilities[0] * 100), 2),
            'Potentially_Habitable': round(float(probabilities[1] * 100), 2),
            'Highly_Habitable': round(float(probabilities[2] * 100), 2)
        }
        
        # Earth similarity (simple calculation)
        earth_similarity = 100.0 if params['radius'] == 1.0 and params['mass'] == 1.0 else 50.0
        
        # Confidence
        if probability_score >= 90:
            confidence = "Very High"
        elif probability_score >= 75:
            confidence = "High"
        elif probability_score >= 60:
            confidence = "Moderate"
        elif probability_score >= 40:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        # Response
        response = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': prediction_label,
            'probability': round(probability_score, 2),
            'probabilities': formatted_probs,
            'earth_similarity': round(earth_similarity, 2),
            'confidence': confidence,
            'model_used': 'XGBoost (Trained Model)'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test connection endpoint"""
    return jsonify({
        'success': True,
        'message': 'Flask server is running!',
        'model_status': 'loaded' if model else 'not loaded'
    })

# ========== MAIN ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌌 ASTROHAB SERVER STARTING")
    print("="*60)
    print(f"📡 Host: {HOST}")
    print(f"🚪 Port: {PORT}")
    print(f"🤖 Model: {'LOADED' if model else 'NOT LOADED'}")
    print("\n🔗 Available URLs:")
    print(f"   🌐 Main Page: http://localhost:{PORT}/")
    print(f"   🩺 Health: http://localhost:{PORT}/api/health")
    print(f"   🔍 Test: http://localhost:{PORT}/api/test")
    print(f"   🤖 Predict: POST http://localhost:{PORT}/api/predict")
    print("\n⚡ Starting server...")
    print("="*60 + "\n")
    

    app.run(host=HOST, port=PORT, debug=False)
