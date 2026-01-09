import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
FEATURES = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_eqt']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
      
        input_data = [data[feat] for feat in FEATURES]
        df_input = pd.DataFrame([input_data], columns=FEATURES)
      
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]  
        
        return jsonify({
            'habitable': bool(prediction),
            'probability': float(probability),
            'features_used': dict(zip(FEATURES, input_data))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
