from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import json

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'models/random_forest_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAZWRI01LSUEWEWb2AfIMrv7f6f0WulTv0')

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load model and scaler
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("Model and scaler loaded successfully")
        else:
            print("Model files not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

def train_model():
    """Train Random Forest model on Kepler data"""
    global model, scaler
    
    # Check if CSV file exists (try multiple possible locations)
    csv_path = '../Kepler_Threshold_Crossing_Events_Table.csv'
    if not os.path.exists(csv_path):
        csv_path = 'Kepler_Threshold_Crossing_Events_Table.csv'
    if not os.path.exists(csv_path):
        print(f"CSV file not found. Tried: ../Kepler_Threshold_Crossing_Events_Table.csv and Kepler_Threshold_Crossing_Events_Table.csv")
        return False
    
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv(csv_path, comment='#', low_memory=False)
        
        # Select relevant features for habitability prediction
        feature_columns = [
            'koi_period',           # Orbital period
            'koi_impact',           # Impact parameter
            'koi_duration',         # Transit duration
            'koi_depth',            # Transit depth
            'koi_prad',             # Planetary radius (Earth radii)
            'koi_teq',              # Equilibrium temperature
            'koi_insol',            # Insolation flux
            'koi_slogg',            # Stellar surface gravity
            'koi_srad',             # Stellar radius
            'koi_steff'             # Stellar effective temperature
        ]
        
        # Filter out rows with missing values in key columns
        available_features = [col for col in feature_columns if col in df.columns]
        df_clean = df[available_features + ['koi_disposition']].dropna()
        
        if len(df_clean) == 0:
            print("No valid data after cleaning")
            return False
        
        X = df_clean[available_features]
        
        # Create target variable: habitability score (0-100)
        # Based on known habitable conditions
        y = np.zeros(len(df_clean))
        
        for idx, row in df_clean.iterrows():
            score = 0
            
            # Temperature score (optimal: 200-320K)
            if 'koi_teq' in available_features and pd.notna(row['koi_teq']):
                temp = row['koi_teq']
                if 200 <= temp <= 320:
                    temp_score = 1.0 - abs(temp - 288) / 88
                    score += max(0, temp_score) * 40
            
            # Radius score (optimal: 0.5-2.5 Earth radii)
            if 'koi_prad' in available_features and pd.notna(row['koi_prad']):
                radius = row['koi_prad']
                if 0.5 <= radius <= 2.5:
                    rad_score = 1.0 - abs(radius - 1.0) / 1.5
                    score += max(0, rad_score) * 30
            
            # Insolation score (optimal: 0.25-2.0)
            if 'koi_insol' in available_features and pd.notna(row['koi_insol']):
                insol = row['koi_insol']
                if 0.25 <= insol <= 2.0:
                    insol_score = 1.0 - abs(insol - 1.0) / 1.0
                    score += max(0, insol_score) * 20
            
            # Period score (optimal: 50-500 days)
            if 'koi_period' in available_features and pd.notna(row['koi_period']):
                period = row['koi_period']
                if 50 <= period <= 500:
                    period_score = 1.0 - abs(period - 365) / 315
                    score += max(0, period_score) * 10
            
            y[idx] = min(100, max(0, score))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        print("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Training R²: {train_score:.4f}")
        print(f"Testing R²: {test_score:.4f}")
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        print("Model trained and saved successfully")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict habitability score using Random Forest model"""
    try:
        data = request.json
        
        # Extract features
        features = {
            'koi_period': data.get('period', 365),
            'koi_prad': data.get('radius', 1.0),
            'koi_teq': data.get('temp', 288),
            'koi_insol': data.get('insol', 1.0),
            'koi_steff': data.get('steff', 5778),
            'koi_srad': data.get('sradius', 1.0),
            'koi_impact': 0.5,  # Default values for missing features
            'koi_duration': 0.1,
            'koi_depth': 0.001,
            'koi_slogg': 4.4
        }
        
        # Convert to array in correct order
        feature_order = [
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_slogg',
            'koi_srad', 'koi_steff'
        ]
        
        feature_array = np.array([[features[key] for key in feature_order]])
        
        # Scale features
        if scaler is None:
            load_model()
        
        if scaler is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        feature_scaled = scaler.transform(feature_array)
        
        # Predict
        if model is None:
            load_model()
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        prediction = model.predict(feature_scaled)[0]
        prediction = max(0, min(100, prediction))  # Clamp between 0 and 100
        
        return jsonify({
            'score': round(prediction, 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare two planets using Gemini API"""
    try:
        data = request.json
        source = data.get('source', 'Earth')
        target = data.get('target', 'Mars')
        
        prompt = f"""Scientifically compare {source} and {target} for human habitability.
        Return detailed JSON including:
        - metrics: {{ temperature, gravity, atmosphere, distance, orbit, radiation }} each with {{ source, target, delta, score }}
        - sourceHabitabilityScore (0-100)
        - targetHabitabilityScore (0-100)
        - status (e.g., 'Habitable', 'Moderate', 'Uninhabitable')
        - winner (planet name)
        - predictionText (one concise summary sentence)
        - detailedAnalysis (extended reasoning 3-4 sentences)
        
        Return ONLY valid JSON, no markdown formatting."""
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "metrics": {
                        "type": "OBJECT",
                        "properties": {
                            "temperature": {"type": "OBJECT", "properties": {"source": {"type": "STRING"}, "target": {"type": "STRING"}, "delta": {"type": "STRING"}, "score": {"type": "NUMBER"}}},
                            "gravity": {"type": "OBJECT", "properties": {"source": {"type": "STRING"}, "target": {"type": "STRING"}, "delta": {"type": "STRING"}, "score": {"type": "NUMBER"}}},
                            "atmosphere": {"type": "OBJECT", "properties": {"source": {"type": "STRING"}, "target": {"type": "STRING"}, "delta": {"type": "STRING"}, "score": {"type": "NUMBER"}}},
                            "distance": {"type": "OBJECT", "properties": {"source": {"type": "STRING"}, "target": {"type": "STRING"}, "delta": {"type": "STRING"}, "score": {"type": "NUMBER"}}},
                            "orbit": {"type": "OBJECT", "properties": {"source": {"type": "STRING"}, "target": {"type": "STRING"}, "delta": {"type": "STRING"}, "score": {"type": "NUMBER"}}},
                            "radiation": {"type": "OBJECT", "properties": {"source": {"type": "STRING"}, "target": {"type": "STRING"}, "delta": {"type": "STRING"}, "score": {"type": "NUMBER"}}}
                        }
                    },
                    "sourceHabitabilityScore": {"type": "NUMBER"},
                    "targetHabitabilityScore": {"type": "NUMBER"},
                    "status": {"type": "STRING"},
                    "winner": {"type": "STRING"},
                    "predictionText": {"type": "STRING"},
                    "detailedAnalysis": {"type": "STRING"}
                }
            }
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        result = json.loads(response.text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    # Try to load existing model
    load_model()
    
    # If model doesn't exist, train it
    if model is None:
        print("Training new model...")
        train_model()
        load_model()
    
    app.run(debug=True, port=5000)
