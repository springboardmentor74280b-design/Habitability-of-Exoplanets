# Exoplanet Habitability Prediction Project

A complete machine learning pipeline to predict and analyze exoplanet habitability using a trained model, REST API, and interactive Streamlit dashboard.

## 📋 Project Overview

This project combines:
- **Machine Learning Model**: Trained habitability prediction model
- **REST API**: Flask-based backend API for predictions
- **Interactive Frontend**: Streamlit web application for exploration and visualization
- **Testing Suite**: Unit tests and helper scripts for validation

## 📁 Project Structure

```
Habitability/
├── data/
│   └── PSCompPars_2025.12.15_07.37.02.csv    # Exoplanet dataset
├── frontend/
│   ├── README.md                              # Frontend-specific documentation
│   ├── streamlit_app.py                       # Streamlit UI
│   └── requirements.txt                       # Frontend dependencies
├── notebook/
│   ├── model.ipynb                            # Model training notebook
│   ├── exo_habitability_final.csv             # Processed dataset for API
│   ├── baseline_models_comparison.csv         # Model comparison results
│   ├── feature_importance_ranking.csv         # Feature importance data
│   ├── sampling_techniques_comparison.csv     # Sampling method comparison
│   ├── Top_Habitable_Exoplanets.csv           # Top habitable planets
│   └── api/
│       ├── app.py                             # Flask REST API
│       ├── requirements.txt                   # API dependencies
│       └── model.pkl                          # Trained model (binary)
├── scripts/
│   ├── check_model.py                         # Model validation script
│   ├── inspect_model.py                       # Model inspection script
│   ├── test_predict.py                        # Python API test script
│   └── test_predict.ps1                       # PowerShell API test script
├── tests/
│   └── test_api.py                            # Pytest suite for API
└── README.md                                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip (or conda)
- Git (optional)

### Installation & Setup

#### 1. **Clone or Download the Project**
```bash
cd Habitability
```

#### 2. **Set Up the Backend API**

Navigate to the API directory and install dependencies:
```bash
cd notebook/api
pip install -r requirements.txt
cd ../..
```

#### 3. **Set Up the Frontend**

Install frontend dependencies:
```bash
cd frontend
pip install -r requirements.txt
cd ..
```

### Running the Application

You'll need to run two components:

#### **Terminal 1: Start the Flask API**
```bash
python notebook/api/app.py
```
The API will start at `http://localhost:5000`

#### **Terminal 2: Start the Streamlit App**
```bash
streamlit run frontend/streamlit_app.py
```
The Streamlit app will open in your browser at `http://localhost:8501`

## 📊 Features

### **Prediction Tab**
- Input exoplanet parameters (HSI, density, temperature, radius, mass, star type, etc.)
- Get real-time habitability predictions
- View prediction confidence and details

### **Top Habitable Planets**
- Explore the most habitable exoplanets from the dataset
- View rankings and key metrics

### **Feature Importance**
- Understand which features most influence habitability predictions
- Visualize feature importance rankings

### **Model & Sampling Comparisons**
- Compare different baseline models
- Analyze sampling techniques performance

### **Dataset Viewer**
- Browse the full exoplanet dataset
- Filter and explore raw data

## 🧪 Testing

### Test the API Directly

**Using Python Script:**
```bash
python scripts/test_predict.py --url http://127.0.0.1:5000
```

**Using PowerShell (Windows):**
```powershell
.\scripts\test_predict.ps1 -Url 'http://127.0.0.1:5000'
```

**Using pytest:**
```bash
pytest tests/test_api.py -v
```
(Requires the API server to be running)

### Validate Model Loading

Check if the model loads correctly:
```bash
python scripts/check_model.py
```

Inspect model details:
```bash
python scripts/inspect_model.py
```

## 📝 Model Information

### Input Features (17 features)
The model expects the following input features:
- `pl_rade`: Planet radius (Earth radii)
- `pl_bmasse`: Planet mass (Earth masses)
- `pl_orbper`: Orbital period (days)
- `pl_orbsmax`: Semi-major axis (AU)
- `pl_eqt`: Equilibrium temperature (K)
- `pl_insol`: Insolation flux (Earth flux)
- `st_teff`: Star effective temperature (K)
- `st_rad`: Star radius (Solar radii)
- `st_met`: Star metallicity
- `planet_density`: Calculated planet density
- `star_luminosity`: Star luminosity (Solar luminosities)
- `HSI`: Habitability Similarity Index
- `SCI`: Solar Compatibility Index
- `star_type_G`, `star_type_K`, `star_type_M`, `star_type_F`: One-hot encoded star types

### Output
Binary classification: Habitable (1) or Not Habitable (0)

## 📚 Training & Data

- **Training Notebook**: [notebook/model.ipynb](notebook/model.ipynb) - Contains the complete model training pipeline
- **Raw Data**: [data/PSCompPars_2025.12.15_07.37.02.csv](data/PSCompPars_2025.12.15_07.37.02.csv)
- **Processed Data**: [notebook/exo_habitability_final.csv](notebook/exo_habitability_final.csv)

Key analysis files:
- [notebook/baseline_models_comparison.csv](notebook/baseline_models_comparison.csv) - Model performance comparison
- [notebook/feature_importance_ranking.csv](notebook/feature_importance_ranking.csv) - Feature importance analysis
- [notebook/sampling_techniques_comparison.csv](notebook/sampling_techniques_comparison.csv) - Sampling method analysis

## 🔧 API Endpoints

### **POST /predict**
Predict habitability for one or multiple exoplanets.

**Request Body** (JSON):
```json
{
  "pl_rade": 1.0,
  "pl_bmasse": 1.0,
  "pl_orbper": 365,
  "pl_orbsmax": 1.0,
  "pl_eqt": 300,
  "pl_insol": 1.0,
  "st_teff": 5800,
  "st_rad": 1.0,
  "st_met": 0.0,
  "planet_density": 5.5,
  "star_luminosity": 1.0,
  "HSI": 0.5,
  "SCI": 0.8,
  "star_type_G": 1,
  "star_type_K": 0,
  "star_type_M": 0,
  "star_type_F": 0
}
```

**Response** (JSON):
```json
{
  "prediction": 1,
  "probability": 0.85,
  "message": "Prediction successful"
}
```

### **GET /health**
Check API status.

**Response**:
```json
{
  "status": "ok"
}
```

## 🛠️ Troubleshooting

### Model Not Found
```
ERROR: model.pkl not found at expected location.
```
**Solution**: Ensure the trained model is saved at `notebook/api/model.pkl`. Run the training notebook first if needed.

### API Connection Error
```
ConnectionError: Failed to connect to http://localhost:5000
```
**Solution**: Make sure the Flask API is running (Terminal 1 command).

### Port Already in Use
```
Address already in use
```
**Solution**: 
- Flask (port 5000): `python notebook/api/app.py --port 5001`
- Streamlit: `streamlit run frontend/streamlit_app.py --server.port 8502`

### Import Errors
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Reinstall dependencies:
```bash
pip install -r frontend/requirements.txt
pip install -r notebook/api/requirements.txt
```

## 📖 Additional Documentation

- [Frontend Documentation](frontend/README.md) - Detailed info on the Streamlit application
- [Training Notebook](notebook/model.ipynb) - Model development and analysis

## 🤝 Development

### Adding New Features
1. Update the model in [notebook/model.ipynb](notebook/model.ipynb)
2. Export the trained model as `model.pkl` to `notebook/api/`
3. Update feature lists in [notebook/api/app.py](notebook/api/app.py) if needed
4. Add frontend changes to [frontend/streamlit_app.py](frontend/streamlit_app.py)
5. Test with the test scripts and pytest suite

### Running Tests
```bash
# Start API first
python notebook/api/app.py

# In another terminal
pytest tests/test_api.py -v
```

## 📄 License

[Add your license information here]

For issues or questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review test outputs from `pytest tests/test_api.py`
3. Run `python scripts/check_model.py` for diagnostics

---

**Last Updated**: January 2026