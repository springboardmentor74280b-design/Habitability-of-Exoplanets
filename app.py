from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from model_utils import apply_physics_engine, predict_habitability, load_model
from dashboard_logic import generate_dashboard_plots, generate_user_plots
from explainability import setup_explainability, explain_prediction

app = Flask(__name__)
# --- GLOBAL CACHE VARIABLE ---
# Stores the images so we don't have to regenerate them every time
DASHBOARD_CACHE = None 

# --- 1. LOAD BRAIN ON STARTUP ---
print("🚀 Server Starting...")
pipeline = load_model()

# --- NEW: Initialize SHAP ---
if pipeline:
    # We pass a dummy dataset or None just to init the TreeExplainer
    setup_explainability(pipeline, None)

if pipeline:
    print("✅ AI Online: Ready for analysis.")
else:
    print("⚠️ WARNING: AI Offline. Did you run train_model.py?")

# --- 2. WEBPAGE ROUTES ---
@app.route('/')
def home():
    """Renders the Marketing/Home page"""
    return render_template('home.html')

@app.route('/analyze')
def analyze():
    """Renders the Tool/Analysis page"""
    return render_template('analyze.html')


# --- 3. DASHBOARD ROUTE (OPTIMIZED) ---
@app.route('/dashboard')
def dashboard_hub():
    """Renders the Global Analytics Dashboard with Caching."""
    global DASHBOARD_CACHE
    
    # Check if model is loaded
    if not pipeline:
        return "<h1>AI Offline. Run train_model.py</h1>"
    
    # 1. Check Cache: Do we already have the images?
    if DASHBOARD_CACHE is None:
        print("⏳ Cache Empty. Generating Dashboard Plots (This happens only once)...")
        try:
            DASHBOARD_CACHE = generate_dashboard_plots()
        except Exception as e:
            print(f"❌ Error generating dashboard: {e}")
            return f"<h1>Error generating dashboard: {e}</h1>"
    else:
        print("⚡ Serving Dashboard from Cache (Instant Load)")
    
    return render_template('dashboard.html', plots=DASHBOARD_CACHE)

@app.route('/dashboard/features')
def show_features():
    return render_template('dashboard_1_feature_importance.html')

@app.route('/dashboard/correlations')
def show_correlations():
    return render_template('dashboard_2_correlations.html')

@app.route('/dashboard/distribution')
def show_distribution():
    return render_template('dashboard_3_score_distribution.html')

# --- NEW: User Plot Route ---
@app.route('/visualize_user_data', methods=['POST'])
def visualize_user_data():
    try:
        req_data = request.get_json()
        
        # Extract data and the limit flag
        raw_data = req_data.get('data', [])
        limit_flag = req_data.get('limit', True) # Default to True (Performance Mode)
        
        df = pd.DataFrame(raw_data)
        
        # Fix column names if needed
        rename_map = {
            'mass': 'P_MASS_EST', 'radius': 'P_RADIUS_EST', 
            'period': 'P_PERIOD', 'distance': 'P_DISTANCE',
            'temp': 'P_TEMP_EQUIL', 'star_temp': 'S_TEMPERATURE',
            'star_lum': 'S_LUMINOSITY', 's_radius': 'S_RADIUS',
            's_mass': 'S_MASS', 's_metallicity': 'S_METALLICITY'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Generate Plots with the User's Limit Choice
        plots = generate_user_plots(df, limit_data=limit_flag)
        
        return jsonify(plots)
    except Exception as e:
        print(f"Plot Error: {e}")
        return jsonify({"error": str(e)}), 500
    
# --- 4. THE API ROUTE (Updated with Robust Handling) ---
@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    """Receives CSV, runs AI, returns results."""
    
    # Check if AI is loaded
    if not pipeline:
        return jsonify({"error": "AI Model is offline. Please run train_model.py first."}), 500

    # Check if file exists in request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # A. Read the uploaded CSV (Handle Python Parser error)
        try:
            user_df = pd.read_csv(file, comment='#') # Handle comments in NASA files
        except:
            file.seek(0)
            user_df = pd.read_csv(file)
        
        # B. Apply Physics Engine (Fix missing data & Map columns)
        processed_df = apply_physics_engine(user_df)
        
        # C. Predict using the loaded AI model
        results = predict_habitability(processed_df, pipeline)
        
        # D. Send JSON back to the browser
        return jsonify(results)

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ... (Previous imports and routes remain the same) ...

# --- NEW: Single Planet Prediction Route (Point 2) ---
@app.route('/predict_single', methods=['POST'])
def predict_single():
    # 1. Check Model
    if not pipeline:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 500
    
    try:
        data = request.get_json()
        
        # 2. Convert to DataFrame & Apply Physics
        # 2. Convert to DataFrame & Apply Physics
        df = pd.DataFrame([data])
        df = apply_physics_engine(df)

        # --- NEW: SOLAR DEFAULTS ---
        # If the user didn't provide Star info, assume a Sun-like star
        # instead of letting it fall to 0 (which confuses the AI).
        if pd.isnull(df['S_TEMPERATURE'].iloc[0]) or df['S_TEMPERATURE'].iloc[0] == 0:
            df['S_TEMPERATURE'] = 5778  # Sun's Temp (K)
            
        if pd.isnull(df['S_MASS'].iloc[0]) or df['S_MASS'].iloc[0] == 0:
            df['S_MASS'] = 1.0  # Sun's Mass (Solar Units)
            
        if pd.isnull(df['S_RADIUS'].iloc[0]) or df['S_RADIUS'].iloc[0] == 0:
            df['S_RADIUS'] = 1.0  # Sun's Radius
            
        if pd.isnull(df['S_LUMINOSITY'].iloc[0]) or df['S_LUMINOSITY'].iloc[0] == 0:
            df['S_LUMINOSITY'] = 1.0 # Sun's Luminosity
            
        if pd.isnull(df['S_METALLICITY'].iloc[0]):
            df['S_METALLICITY'] = 0.0 # Sun's Metallicity (Log scale)
        # ---------------------------
        
        # 3. Predict
        features = ['P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
                    'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 
                    'S_RADIUS', 'S_MASS', 'S_METALLICITY']
        
        X = df[features].fillna(0)
        
        # Get Verdict
        prediction_code = pipeline.predict(X)[0]
        labels = {0: 'Non-Habitable', 1: 'Habitable', 2: 'Optimistic'}
        verdict = labels.get(prediction_code, "Unknown")
        
        # Get Probability/Score
        probs = pipeline.predict_proba(X)[0]
        if len(probs) == 3:
            score = (probs[1] + probs[2]) * 100
        else:
            score = probs[1] * 100

        # 4. Get SHAP Explanation (Safely)
        try:
            reasons = explain_prediction(pipeline, df)
        except Exception as e:
            print(f"⚠️ Warning: SHAP failed: {e}")
            reasons = [] # Fallback to empty list if SHAP crashes
            
        # 5. Return JSON (With explicit type casting to fix serialization errors)
        return jsonify({
            'prediction': verdict,
            'probability': float(score) / 100, 
            'score': round(float(score), 1),
            'reasons': reasons,
            # Safe Casting for Stats
            'temp': float(df['P_TEMP_EQUIL'].iloc[0]) if pd.notnull(df['P_TEMP_EQUIL'].iloc[0]) else None,
            'radius': float(df['P_RADIUS_EST'].iloc[0]) if pd.notnull(df['P_RADIUS_EST'].iloc[0]) else None,
            'period': float(df['P_PERIOD'].iloc[0]) if pd.notnull(df['P_PERIOD'].iloc[0]) else None,
            'star_temp': float(df['S_TEMPERATURE'].iloc[0]) if pd.notnull(df['S_TEMPERATURE'].iloc[0]) else None
        })

    except Exception as e:
        print(f"❌ Error in predict_single: {e}")
        # This return ensures the server doesn't crash with 'did not return a valid response'
        return jsonify({'error': str(e)}), 500# ... (End of file: if __name__ == '__main__': app.run...)

if __name__ == '__main__':
    # Get the PORT from the environment (Render sets this automatically)
    # Default to 5000 if running locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)