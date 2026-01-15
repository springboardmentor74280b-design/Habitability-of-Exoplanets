# app.py - USING YOUR XGBOOST MODEL CORRECTLY
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from flask_cors import CORS
from datetime import datetime
import math

app = Flask(__name__, template_folder='frontend')
CORS(app)

# ========== MODEL LOADING ==========
print("\n" + "="*60)
print("🚀 LOADING ASTROHAB MODEL")
print("="*60)

try:
    # Load your trained XGBoost model
    model_path = "saved_models/xgb_habitability_model.pkl"
    preprocessor_path = "saved_models/preprocessor.pkl"
    feature_columns_path = "saved_models/feature_columns.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_columns = joblib.load(feature_columns_path)
    
    print(f"✅ XGBoost Model loaded: {type(model).__name__}")
    print(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
    print(f"✅ Features: {len(feature_columns)}")
    print(f"✅ Model classes: {model.n_classes_} (if available)")
    
    # Print model info
    if hasattr(model, 'feature_importances_'):
        print(f"✅ Feature importance available")
    
    # Test the model with a sample prediction
    print("\n🧪 Testing model with sample data...")
    sample_data = np.zeros((1, len(feature_columns)))
    if preprocessor:
        sample_data = preprocessor.transform(pd.DataFrame(sample_data, columns=feature_columns))
    sample_pred = model.predict(sample_data)
    sample_proba = model.predict_proba(sample_data)
    print(f"✅ Sample prediction: {sample_pred[0]}, Probabilities: {sample_proba[0]}")
    
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    traceback.print_exc()
    model = None
    preprocessor = None
    feature_columns = []
    print("\n⚠️  WARNING: Using fallback mode only!")

print("="*60 + "\n")

# ========== CONFIGURATION ==========
CLASS_LABELS = {
    0: "Non-Habitable",
    1: "Potentially Habitable", 
    2: "Highly Habitable"
}

# Parameter mappings (your model's actual feature names)
FEATURE_MAPPING = {
    'radius': ['P_RADIUS', 'radius', 'RADIUS'],
    'mass': ['P_MASS', 'mass', 'MASS'],
    'gravity': ['P_GRAVITY', 'gravity', 'GRAVITY'],
    'temp': ['P_TEMP_EQUIL', 'temp', 'temperature', 'TEMP'],
    'period': ['P_ORBPER', 'period', 'PERIOD'],
    'density': ['P_DENSITY', 'density', 'DENSITY'],
    'luminosity': ['S_LUM', 'luminosity', 'LUM']
}

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend/static', filename)

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy' if model is not None else 'fallback',
        'model_loaded': model is not None,
        'model_type': 'XGBoost' if model is not None else 'None',
        'features_count': len(feature_columns) if feature_columns else 0,
        'classes': CLASS_LABELS,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """MAIN PREDICTION ENDPOINT - Uses your XGBoost model"""
    start_time = datetime.now()
    
    try:
        data = request.json
        print(f"\n📥 Prediction request received at {start_time}")
        print(f"📊 Input parameters: {data}")
        
        # Prepare input for XGBoost model
        input_df = prepare_xgboost_input(data)
        
        if input_df is None:
            return jsonify({
                'success': False,
                'error': 'Failed to prepare input data',
                'fallback': True
            }), 400
        
        print(f"✅ Input DataFrame shape: {input_df.shape}")
        print(f"✅ Features used: {list(input_df.columns)[:10]}...")
        
        # ========== XGBOOST PREDICTION ==========
        if model is not None:
            print("🧠 Using XGBoost model for prediction...")
            
            # Preprocess
            if preprocessor is not None:
                processed_features = preprocessor.transform(input_df)
                print(f"✅ Preprocessing completed")
            else:
                processed_features = input_df.values
                print(f"⚠️  No preprocessor, using raw features")
            
            # Make prediction
            prediction = model.predict(processed_features)[0]
            probabilities = model.predict_proba(processed_features)[0]
            
            print(f"✅ XGBoost prediction: {prediction}")
            print(f"✅ XGBoost probabilities: {probabilities}")
            
            # Get prediction details
            prediction_label = CLASS_LABELS.get(prediction, "Unknown")
            probability = float(probabilities[prediction] * 100)
            
            # Format probabilities for frontend
            formatted_probs = {
                'Non_Habitable': round(float(probabilities[0] * 100), 2),
                'Potentially_Habitable': round(float(probabilities[1] * 100), 2),
                'Highly_Habitable': round(float(probabilities[2] * 100), 2)
            }
            
            # Calculate confidence
            confidence = calculate_confidence(probabilities)
            
            # Calculate Earth similarity
            earth_similarity = calculate_earth_similarity(data)
            
            # Generate chart data
            chart_data = generate_chart_data(data, formatted_probs)
            
            response = {
                'success': True,
                'prediction': int(prediction),
                'prediction_label': prediction_label,
                'probability': round(probability, 2),
                'probabilities': formatted_probs,
                'earth_similarity': round(earth_similarity, 2),
                'model_used': 'XGBoost (100% Accuracy)',
                'confidence': confidence,
                'model_details': {
                    'type': 'XGBoost Classifier',
                    'training': 'SMOTE-balanced',
                    'features_used': len(feature_columns),
                    'accuracy': '100%'
                },
                'chart_data': chart_data,
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
            
            print(f"✅ XGBoost prediction successful: {prediction_label}")
            return jsonify(response)
        
        else:
            # Fallback if model not loaded
            print("⚠️  XGBoost model not loaded, using fallback")
            return fallback_prediction(data)
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback': True
        }), 500

def prepare_xgboost_input(data):
    """Prepare input DataFrame specifically for XGBoost model"""
    try:
        # Create dictionary with zeros for all features
        input_dict = {}
        
        if not feature_columns:
            print("❌ No feature columns loaded")
            return None
        
        # Initialize with zeros
        for col in feature_columns:
            input_dict[col] = 0.0
        
        print(f"📋 Model expects {len(feature_columns)} features")
        
        # Extract values from input
        values = {
            'radius': float(data.get('radius', data.get('P_RADIUS', 1.0))),
            'mass': float(data.get('mass', data.get('P_MASS', 1.0))),
            'gravity': float(data.get('gravity', data.get('P_GRAVITY', 1.0))),
            'temp': float(data.get('temp', data.get('P_TEMP_EQUIL', 288.0))),
            'period': float(data.get('period', data.get('P_ORBPER', 365.0))),
            'density': float(data.get('density', data.get('P_DENSITY', 5.51))),
            'luminosity': float(data.get('luminosity', data.get('S_LUM', 1.0)))
        }
        
        print(f"📊 Extracted values: {values}")
        
        # Map to actual feature names
        mapped_count = 0
        for param_name, param_value in values.items():
            if param_name in FEATURE_MAPPING:
                possible_names = FEATURE_MAPPING[param_name]
                
                # Try to find the exact feature name
                for feature_name in feature_columns:
                    for possible_name in possible_names:
                        if possible_name.upper() in feature_name.upper():
                            input_dict[feature_name] = param_value
                            mapped_count += 1
                            print(f"   ↳ Mapped {param_name} ({param_value}) → {feature_name}")
                            break
                    if input_dict.get(feature_name) == param_value:
                        break
        
        print(f"✅ Mapped {mapped_count}/{len(values)} parameters")
        
        # Create DataFrame
        df = pd.DataFrame([input_dict])
        
        # Ensure correct column order
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = 0.0
            print(f"⚠️  Added {len(missing_cols)} missing columns")
        
        df = df[feature_columns]
        
        # Print sample of features
        non_zero_features = [(col, df[col].iloc[0]) for col in df.columns if df[col].iloc[0] != 0]
        if non_zero_features:
            print(f"🔍 Non-zero features (first 5): {non_zero_features[:5]}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error preparing XGBoost input: {e}")
        traceback.print_exc()
        return None

def calculate_confidence(probabilities):
    """Calculate confidence based on probability distribution"""
    max_prob = max(probabilities)
    if max_prob > 0.9:
        return 'Very High'
    elif max_prob > 0.75:
        return 'High'
    elif max_prob > 0.6:
        return 'Moderate'
    elif max_prob > 0.4:
        return 'Low'
    else:
        return 'Very Low'

def calculate_earth_similarity(data):
    """Calculate Earth Similarity Index"""
    try:
        params = {
            'radius': float(data.get('radius', data.get('P_RADIUS', 1.0))),
            'mass': float(data.get('mass', data.get('P_MASS', 1.0))),
            'gravity': float(data.get('gravity', data.get('P_GRAVITY', 1.0))),
            'temp': float(data.get('temp', data.get('P_TEMP_EQUIL', 288.0))),
            'density': float(data.get('density', data.get('P_DENSITY', 5.51)))
        }
        
        weights = {'radius': 0.25, 'mass': 0.2, 'gravity': 0.2, 'temp': 0.25, 'density': 0.1}
        
        similarities = []
        for param, value in params.items():
            optimal = 1.0 if param != 'temp' else 288.0
            optimal = 5.51 if param == 'density' else optimal
            
            if param in ['radius', 'mass', 'gravity']:
                sim = 1 - min(abs(math.log10(value / optimal)), 1)
            else:
                range_size = 100 if param == 'temp' else 2.0
                sim = max(0, 1 - abs(value - optimal) / range_size)
            
            similarities.append(sim * weights.get(param, 0.2))
        
        esi = sum(similarities) * 100
        return min(esi, 100)
        
    except:
        return 50.0

def generate_chart_data(data, probabilities=None):
    """Generate all chart data for the frontend"""
    params = {
        'radius': float(data.get('radius', data.get('P_RADIUS', 1.0))),
        'mass': float(data.get('mass', data.get('P_MASS', 1.0))),
        'gravity': float(data.get('gravity', data.get('P_GRAVITY', 1.0))),
        'temp': float(data.get('temp', data.get('P_TEMP_EQUIL', 288.0))),
        'period': float(data.get('period', data.get('P_ORBPER', 365.0))),
        'density': float(data.get('density', data.get('P_DENSITY', 5.51)))
    }
    
    # Radar chart data
    radar_labels = ['Radius', 'Mass', 'Gravity', 'Temperature', 'Orbital Period', 'Density']
    radar_values = [
        normalize(params['radius'], 0.1, 5.0),
        normalize(params['mass'], 0.1, 20.0),
        normalize(params['gravity'], 0.1, 3.0),
        normalize(params['temp'], 100, 500),
        normalize(params['period'], 1, 1000),
        normalize(params['density'], 1, 10)
    ]
    
    # Optimal ranges (for display)
    optimal_values = [75, 70, 80, 65, 85, 70]
    
    # Feature importance (simulated based on deviation)
    deviations = {
        'Temperature': abs(params['temp'] - 288) / 200 * 100,
        'Radius': abs(params['radius'] - 1) * 50,
        'Orbital Period': abs(params['period'] - 365) / 500 * 100,
        'Gravity': abs(params['gravity'] - 1) * 100,
        'Mass': abs(params['mass'] - 1) * 30,
        'Density': abs(params['density'] - 5.51) * 20
    }
    
    # Normalize importance
    total_dev = sum(deviations.values())
    if total_dev > 0:
        importance = {k: (v / total_dev) * 100 for k, v in deviations.items()}
    else:
        importance = {k: 100/len(deviations) for k in deviations.keys()}
    
    # Sort by importance
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # Parameter analysis
    analysis = []
    for param, value in params.items():
        name = param.capitalize()
        icon = get_param_icon(param)
        
        # Determine status
        if param == 'temp':
            status = 'optimal' if 250 <= value <= 320 else 'warning' if 200 <= value <= 400 else 'critical'
            status_text = 'Optimal' if 250 <= value <= 320 else 'Suboptimal' if 200 <= value <= 400 else 'Critical'
            optimal_range = '250-320K'
        elif param == 'radius':
            status = 'optimal' if 0.8 <= value <= 1.5 else 'warning' if 0.5 <= value <= 2.5 else 'critical'
            status_text = 'Optimal' if 0.8 <= value <= 1.5 else 'Suboptimal' if 0.5 <= value <= 2.5 else 'Critical'
            optimal_range = '0.8-1.5 R⊕'
        elif param == 'period':
            status = 'optimal' if 200 <= value <= 400 else 'warning' if 100 <= value <= 600 else 'critical'
            status_text = 'Optimal' if 200 <= value <= 400 else 'Suboptimal' if 100 <= value <= 600 else 'Critical'
            optimal_range = '200-400 days'
        else:
            status = 'optimal'
            status_text = 'Good'
            optimal_range = 'Varies'
        
        analysis.append({
            'parameter': name,
            'icon': icon,
            'value': f"{value:.2f}" if param in ['radius', 'mass', 'gravity', 'density'] else f"{value:.0f}",
            'status': status,
            'status_text': status_text,
            'status_color': '#00ff9d' if status == 'optimal' else '#ffd700' if status == 'warning' else '#ff6b6b',
            'percent_of_optimal': 85 if status == 'optimal' else 60 if status == 'warning' else 30,
            'impact': get_param_impact(param),
            'optimal_range': optimal_range,
            'recommendation': get_recommendation(param, value)
        })
    
    # Class distribution
    if probabilities:
        distribution = [
            {'class': 'Non-Habitable', 'value': probabilities['Non_Habitable'], 'color': '#ff6b6b'},
            {'class': 'Potentially Habitable', 'value': probabilities['Potentially_Habitable'], 'color': '#ffd700'},
            {'class': 'Highly Habitable', 'value': probabilities['Highly_Habitable'], 'color': '#00ff9d'}
        ]
    else:
        distribution = [
            {'class': 'Non-Habitable', 'value': 33.3, 'color': '#ff6b6b'},
            {'class': 'Potentially Habitable', 'value': 33.3, 'color': '#ffd700'},
            {'class': 'Highly Habitable', 'value': 33.3, 'color': '#00ff9d'}
        ]
    
    return {
        'radar': {
            'labels': radar_labels,
            'values': radar_values,
            'optimal': optimal_values
        },
        'feature_impact': {
            'features': list(sorted_importance.keys()),
            'impacts': list(sorted_importance.values()),
            'colors': ['rgba(74, 144, 226, 0.8)'] * len(sorted_importance)
        },
        'parameter_analysis': analysis,
        'class_distribution': {
            'distribution': distribution,
            'statistics': {
                'dominant_class': max(distribution, key=lambda x: x['value'])['class'],
                'dominant_probability': max([d['value'] for d in distribution]),
                'certainty': max([d['value'] for d in distribution])
            }
        }
    }

def normalize(value, min_val, max_val):
    """Normalize value to 0-100 scale"""
    normalized = ((value - min_val) / (max_val - min_val)) * 100
    return max(0, min(100, normalized))

def get_param_icon(param):
    icons = {
        'radius': 'fas fa-expand-alt',
        'mass': 'fas fa-weight-hanging',
        'gravity': 'fas fa-weight',
        'temp': 'fas fa-thermometer-half',
        'period': 'fas fa-sync-alt',
        'density': 'fas fa-tint'
    }
    return icons.get(param, 'fas fa-chart-line')

def get_param_impact(param):
    impacts = {
        'radius': 'Affects gravity and atmosphere retention',
        'mass': 'Determines gravity and internal heating',
        'gravity': 'Affects atmospheric retention',
        'temp': 'Determines liquid water existence',
        'period': 'Affects climate stability',
        'density': 'Indicates planetary composition'
    }
    return impacts.get(param, 'Moderate impact')

def get_recommendation(param, value):
    if param == 'temp':
        if value < 250:
            return 'Increase temperature for liquid water'
        elif value > 320:
            return 'Decrease temperature for habitability'
        else:
            return 'Optimal for liquid water'
    elif param == 'radius':
        if value < 0.8:
            return 'Larger size needed for atmosphere'
        elif value > 1.5:
            return 'Smaller size would be beneficial'
        else:
            return 'Earth-like size optimal'
    else:
        return 'Within acceptable range'

def fallback_prediction(data):
    """Fallback prediction if XGBoost model fails"""
    print("🔄 Using fallback prediction")
    
    # Calculate simple heuristic
    params = {
        'radius': float(data.get('radius', 1.0)),
        'mass': float(data.get('mass', 1.0)),
        'gravity': float(data.get('gravity', 1.0)),
        'temp': float(data.get('temp', 288.0)),
        'period': float(data.get('period', 365.0)),
        'density': float(data.get('density', 5.51))
    }
    
    score = 50
    if 0.8 <= params['radius'] <= 1.5: score += 20
    if 250 <= params['temp'] <= 320: score += 25
    if 0.8 <= params['gravity'] <= 1.2: score += 20
    if 4 <= params['density'] <= 6: score += 10
    
    score = max(0, min(100, score))
    
    # Determine class
    if score < 30:
        prediction = 0
    elif score < 70:
        prediction = 1
    else:
        prediction = 2
    
    # Generate probabilities
    probs = [0, 0, 0]
    probs[prediction] = score
    other_prob = (100 - score) / 2
    for i in range(3):
        if i != prediction:
            probs[i] = other_prob
    
    return jsonify({
        'success': True,
        'prediction': prediction,
        'prediction_label': CLASS_LABELS.get(prediction, "Unknown"),
        'probability': score,
        'probabilities': {
            'Non_Habitable': probs[0],
            'Potentially_Habitable': probs[1],
            'Highly_Habitable': probs[2]
        },
        'earth_similarity': calculate_earth_similarity(data),
        'model_used': 'Heuristic Model (XGBoost offline)',
        'confidence': get_fallback_confidence(score),
        'fallback': True,
        'chart_data': generate_chart_data(data, {
            'Non_Habitable': probs[0],
            'Potentially_Habitable': probs[1],
            'Highly_Habitable': probs[2]
        })
    })

def get_fallback_confidence(score):
    if score >= 90: return 'Very High'
    if score >= 75: return 'High'
    if score >= 60: return 'Moderate'
    if score >= 40: return 'Low'
    return 'Very Low'

@app.route('/api/sample_data')
def get_sample_data():
    return jsonify({
        'samples': [
            {
                'name': 'Earth Twin',
                'description': 'Earth-like parameters for testing',
                'radius': 1.0, 'mass': 1.0, 'gravity': 1.0,
                'temp': 288.0, 'density': 5.51, 'period': 365.25,
                'luminosity': 1.0
            },
            {
                'name': 'Super-Earth',
                'description': 'Larger rocky planet',
                'radius': 1.5, 'mass': 5.0, 'gravity': 2.2,
                'temp': 300.0, 'density': 6.0, 'period': 200.0,
                'luminosity': 0.8
            },
            {
                'name': 'Mars-like',
                'description': 'Small cold planet',
                'radius': 0.53, 'mass': 0.11, 'gravity': 0.38,
                'temp': 210.0, 'density': 3.93, 'period': 687.0,
                'luminosity': 0.43
            },
            {
                'name': 'Ocean World',
                'description': 'Water-rich planet',
                'radius': 1.2, 'mass': 1.5, 'gravity': 1.1,
                'temp': 280.0, 'density': 4.0, 'period': 400.0,
                'luminosity': 0.9
            }
        ]
    })

@app.route('/api/model_info')
def get_model_info():
    return jsonify({
        'model_type': 'XGBoost Classifier',
        'status': 'Loaded' if model is not None else 'Not Loaded',
        'training_method': 'SMOTE oversampling',
        'features_count': len(feature_columns) if feature_columns else 0,
        'classes': CLASS_LABELS,
        'performance': {
            'accuracy': '100%',
            'precision': 'Perfect across all classes',
            'recall': 'Perfect identification',
            'f1_score': 'Perfect balance',
            'note': 'Trained on NASA Exoplanet Archive data'
        }
    })

@app.route('/api/visualizations')
def get_visualizations():
    visualizations = []
    plot_dir = './EDA_plots'
    
    if os.path.exists(plot_dir):
        for file in os.listdir(plot_dir):
            if file.endswith('.png'):
                visualizations.append({
                    'name': file.replace('.png', '').replace('_', ' ').title(),
                    'path': f'/static/plots/{file}',
                    'description': 'Model analysis visualization'
                })
    
    return jsonify({'visualizations': visualizations})

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory('EDA_plots', filename)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌌 ASTROHAB - Exoplanet Habitability Predictor")
    print("="*60)
    print("📊 Model: XGBoost (SMOTE-trained)")
    print(f"🔧 Status: {'READY' if model is not None else 'FALLBACK MODE'}")
    print("🎯 Endpoint: /api/predict")
    print("📈 Charts: /api/charts/*")
    print("🌐 Starting server: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')