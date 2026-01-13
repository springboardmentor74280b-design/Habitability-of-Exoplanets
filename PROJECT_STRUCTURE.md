# Project Structure

This document describes the complete project structure for ExoHabitAI.

## Directory Layout

```
internship/
│
├── frontend/                          # Frontend React Application
│   ├── index.html                     # Main HTML entry point
│   ├── app.js                         # React application logic (all components)
│   └── styles.css                     # Custom CSS styles and animations
│
├── backend/                           # Backend Flask API Server
│   ├── app.py                         # Main Flask application with API endpoints
│   ├── train_model.py                 # Standalone script to train ML model
│   ├── requirements.txt               # Python dependencies
│   └── models/                        # Generated directory for saved models
│       ├── random_forest_model.pkl   # Trained Random Forest model (generated)
│       └── scaler.pkl                 # Feature scaler (generated)
│
├── Kepler_Threshold_Crossing_Events_Table.csv  # Dataset (NASA Kepler data)
│
├── README.md                          # Main project documentation
├── SETUP.md                           # Quick setup guide
├── PROJECT_STRUCTURE.md              # This file
└── .gitignore                         # Git ignore rules

```

## File Descriptions

### Frontend Files

- **index.html**: Main HTML structure with CDN links for React, Tailwind CSS, Babel, and FontAwesome
- **app.js**: Complete React application including:
  - Planet data and constants
  - API integration functions
  - All React components (CreatePlanetView, CompareView, ExploreSpaceView, etc.)
  - Main App component with routing
- **styles.css**: Custom animations and styling (fade-in, slide-in, scrollbar styling)

### Backend Files

- **app.py**: Flask server with three main endpoints:
  - `/api/predict` - Random Forest prediction endpoint
  - `/api/compare` - Gemini API comparison endpoint
  - `/api/health` - Health check endpoint
- **train_model.py**: Standalone script to train the Random Forest model
  - Loads Kepler dataset
  - Preprocesses data
  - Trains Random Forest regressor
  - Saves model and scaler
- **requirements.txt**: Python package dependencies

### Configuration Files

- **README.md**: Complete project documentation
- **SETUP.md**: Step-by-step setup instructions
- **.gitignore**: Git ignore patterns for Python, models, and environment files

## API Endpoints

### POST /api/predict
**Purpose**: Predict habitability score using Random Forest model

**Input**:
```json
{
  "radius": 1.0,      // Planetary radius (Earth = 1.0)
  "temp": 288,        // Equilibrium temperature (Kelvin)
  "insol": 1.0,       // Insolation flux (Earth = 1.0)
  "period": 365,      // Orbital period (days)
  "steff": 5778,      // Stellar temperature (Kelvin)
  "sradius": 1.0      // Stellar radius (Sun = 1.0)
}
```

**Output**:
```json
{
  "score": 85.5,      // Habitability score (0-100)
  "status": "success"
}
```

### POST /api/compare
**Purpose**: Compare two planets using Gemini API

**Input**:
```json
{
  "source": "Earth",
  "target": "Mars"
}
```

**Output**: Detailed comparison JSON with metrics, scores, and analysis

### GET /api/health
**Purpose**: Check server and model status

**Output**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

## Technology Stack

### Frontend
- React 18 (via CDN)
- Tailwind CSS (via CDN)
- Babel Standalone (for JSX transformation)
- FontAwesome 6.4.0

### Backend
- Flask 3.0.0
- Flask-CORS 4.0.0
- scikit-learn 1.3.2 (Random Forest)
- pandas 2.0.3
- numpy 1.24.3
- google-generativeai 0.3.2

## Model Details

### Random Forest Configuration
- **Algorithm**: RandomForestRegressor
- **n_estimators**: 100 trees
- **max_depth**: 20
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **Target**: Habitability score (0-100)

### Features Used
1. Orbital period (koi_period)
2. Planetary radius (koi_prad)
3. Equilibrium temperature (koi_teq)
4. Insolation flux (koi_insol)
5. Stellar temperature (koi_steff)
6. Stellar radius (koi_srad)
7. Additional transit parameters

### Scoring Weights
- Temperature: 40%
- Radius: 30%
- Insolation: 20%
- Orbital Period: 10%

## Development Workflow

1. **Train Model**: Run `python backend/train_model.py`
2. **Start Backend**: Run `python backend/app.py`
3. **Start Frontend**: Serve `frontend/index.html` via HTTP server
4. **Test**: Use frontend UI or curl commands (see SETUP.md)

## Notes

- The frontend uses CDN-hosted libraries, so no build step is required
- Model files are generated during training and should be committed to version control (or added to .gitignore if too large)
- Gemini API key must be set as environment variable
- CSV dataset should be in parent directory or backend directory
