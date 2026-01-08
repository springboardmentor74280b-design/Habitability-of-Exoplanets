from flask import Blueprint, render_template, request, jsonify, redirect, url_for
import pandas as pd
import os
from . import db
from .models import Exoplanet
from .services import predict_single, process_csv, analyze_csv_data

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Manual Form Submission
        # Collect data inputs
        # We need to map form fields to model features (P_MASS, P_RADIUS, etc.)
        data = {
            'P_MASS': float(request.form.get('P_MASS', 0)),
            'P_RADIUS': float(request.form.get('P_RADIUS', 0)),
            'P_FLUX': float(request.form.get('P_FLUX', 0)),
            'P_PERIOD': float(request.form.get('P_PERIOD', 0)),
            'P_SEMI_MAJOR_AXIS': float(request.form.get('P_SEMI_MAJOR_AXIS', 0)),
            'P_GRAVITY': float(request.form.get('P_GRAVITY', 0)),
            'S_TEMPERATURE': float(request.form.get('S_TEMPERATURE', 0)),
            'S_MASS': float(request.form.get('S_MASS', 0)),
            'S_RADIUS': float(request.form.get('S_RADIUS', 0)),
        }
        
        result = predict_single(data)
        return render_template('result.html', result=result)
    
    return render_template('predict.html')

@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Process & Save
            df_result = process_csv(file)
            
            # Save to instance folder for pagination
            save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../instance/latest_results.csv')
            df_result.to_csv(save_path, index=False)
            
            return redirect(url_for('main.upload_results'))
            
    return render_template('upload.html')

@main.route('/upload/results')
def upload_results():
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../instance/latest_results.csv')
    
    if not os.path.exists(save_path):
        return redirect(url_for('main.upload'))
        
    # Validation: Check if empty
    try:
        df = pd.read_csv(save_path)
    except:
        return redirect(url_for('main.upload'))

    # Pagination Logic
    page = request.args.get('page', 1, type=int)
    per_page = 20
    total = len(df)
    pages = (total + per_page - 1) // per_page
    
    # Slice
    start = (page - 1) * per_page
    end = start + per_page
    records = df.iloc[start:end].to_dict('records')
    columns = df.columns.tolist()
    
    # Simple pagination object to match dataset.html logic (if used) or custom
    # Let's pass 'has_prev', 'has_next', 'prev_num', 'next_num', 'page', 'pages'
    pagination = {
        'has_prev': page > 1,
        'has_next': page < pages,
        'prev_num': page - 1,
        'next_num': page + 1,
        'page': page,
        'pages': pages,
        'total': total,
        'iter_pages': range(1, pages + 1) if pages < 10 else [1, 2, '...', pages-1, pages] # Simplified
    }
    # Better to use a simpler iter_pages logic for template or just Simple logic
    
    return render_template('results_table.html', records=records, columns=columns, pagination=pagination)

@main.route('/dataset')
def dataset():
    # Helper to get data from DB
    page = request.args.get('page', 1, type=int)
    pagination = Exoplanet.query.paginate(page=page, per_page=20)
    return render_template('dataset.html', pagination=pagination)

@main.route('/api/search_planet')
def search_planet():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    
    # Search in DB
    results = Exoplanet.query.filter(Exoplanet.P_NAME.ilike(f'%{q}%')).limit(10).all()
    
    # Return JSON
    data = []
    for p in results:
        data.append({
            'name': p.P_NAME,
            'mass': p.P_MASS,
            'radius': p.P_RADIUS,
            'flux': p.P_FLUX,
            'period': p.P_PERIOD,
            'distance': p.P_SEMI_MAJOR_AXIS,
            'gravity': p.P_GRAVITY,
            'star_temp': p.S_TEMPERATURE,
            'star_mass': p.S_MASS,
            'star_radius': p.S_RADIUS
        })
    return jsonify(data)

@main.route('/visualize')
def visualize():
    return render_template('visualize.html')

@main.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Expects JSON input: { "P_MASS": 1.0, ... }
    Returns JSON output: { "class": 2, "label": "Habitable" }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No input data provided"}), 400
        
    try:
        result = predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/api/ranking')
def api_ranking():
    """
    Returns top habitable planets from the database.
    Since we don't have a continuous 'habitability score' stored, 
    we return the confirmed 'Very Habitable' (Class 2) planets.
    """
    # Get all Class 2 planets
    habitable_planets = Exoplanet.query.filter_by(P_HABITABLE=2).limit(50).all()
    
    data = []
    for p in habitable_planets:
        data.append({
            'name': p.P_NAME,
            'mass': p.P_MASS,
            'radius': p.P_RADIUS,
            'temp': p.P_TEMP_SURF,
            'habitability_class': p.P_HABITABLE
        })
    
    # Return as JSON
    # Return as JSON
    return jsonify(data)

@main.route('/download_results')
def download_results():
    try:
        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../instance/latest_results.csv')
        if not os.path.exists(save_path):
            return "No results found to export. Please upload a CSV first.", 404
            
        from flask import send_file
        return send_file(save_path, as_attachment=True, download_name='habitability_results.csv')
    except Exception as e:
        return str(e), 500

@main.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Endpoint to analyze uploaded CSV and return stats for visualization.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    try:
        stats = analyze_csv_data(file)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
