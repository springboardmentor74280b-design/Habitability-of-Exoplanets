# ExoHabitAI - Planetary Habitability Prediction System

A full-stack web application that uses Machine Learning (Random Forest) and AI (Google Gemini) to predict and compare planetary habitability scores.

## Project Structure

```
internship/
├── frontend/              # Frontend React application
│   ├── index.html        # Main HTML file
│   ├── app.js            # React application logic
│   └── styles.css        # Custom styles
├── backend/              # Backend Flask API
│   ├── app.py            # Flask server with API endpoints
│   ├── train_model.py    # Model training script
│   ├── models/           # Trained model files (generated)
│   │   ├── random_forest_model.pkl
│   │   └── scaler.pkl
│   └── requirements.txt  # Python dependencies
└── README.md             # This file
```

## Features

1. **Planet Creation & Prediction**: Input planetary parameters and get habitability predictions using a trained Random Forest ML model
2. **Planet Comparison**: Compare two planets using Google Gemini API for detailed scientific analysis
3. **Interactive UI**: Modern, responsive interface built with React and Tailwind CSS
4. **Data Visualization**: Explore analysis visualizations and planetary data

## Prerequisites

- Python 3.8 or higher
- Node.js (optional, for local development)
- Google Gemini API Key (for planet comparison feature)

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# On Windows (PowerShell):
$env:GEMINI_API_KEY="your_gemini_api_key_here"

# On Linux/Mac:
export GEMINI_API_KEY="your_gemini_api_key_here"
```

Or create a `.env` file in the backend directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Train the Random Forest model:
```bash
python train_model.py
```

This will:
- Load the Kepler dataset from the parent directory
- Train a Random Forest model
- Save the model and scaler to `models/` directory

### Frontend Setup

The frontend uses CDN-hosted libraries, so no build step is required. Simply open `frontend/index.html` in a web browser or serve it using a local server.

For development, you can use Python's HTTP server:
```bash
cd frontend
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Running the Application

### 1. Start the Backend Server

```bash
cd backend
python app.py
```

The server will start on `http://localhost:5000`

### 2. Open the Frontend

Open `frontend/index.html` in your web browser, or if using a local server:
```
http://localhost:8000
```

**Note**: Make sure to update the `API_BASE_URL` in `frontend/app.js` if your backend is running on a different port.

## API Endpoints

### `POST /api/predict`
Predict habitability score for custom planetary parameters.

**Request Body:**
```json
{
  "radius": 1.0,
  "temp": 288,
  "insol": 1.0,
  "period": 365,
  "steff": 5778,
  "sradius": 1.0
}
```

**Response:**
```json
{
  "score": 85.5,
  "status": "success"
}
```

### `POST /api/compare`
Compare two planets using Gemini API.

**Request Body:**
```json
{
  "source": "Earth",
  "target": "Mars"
}
```

**Response:**
```json
{
  "metrics": {...},
  "sourceHabitabilityScore": 100,
  "targetHabitabilityScore": 45,
  "status": "Moderate",
  "winner": "Earth",
  "predictionText": "...",
  "detailedAnalysis": "..."
}
```

### `GET /api/health`
Health check endpoint.

## Model Details

The Random Forest model uses the following features:
- `koi_period`: Orbital period (days)
- `koi_prad`: Planetary radius (Earth radii)
- `koi_teq`: Equilibrium temperature (Kelvin)
- `koi_insol`: Insolation flux (Earth = 1.0)
- `koi_steff`: Stellar effective temperature (Kelvin)
- `koi_srad`: Stellar radius (Sun = 1.0)
- Additional features: impact parameter, transit duration, depth, stellar surface gravity

The model predicts a habitability score (0-100) based on:
- Temperature suitability (40% weight)
- Planetary radius (30% weight)
- Insolation flux (20% weight)
- Orbital period (10% weight)

## Technologies Used

- **Frontend**: React, Tailwind CSS, FontAwesome
- **Backend**: Flask, Flask-CORS
- **Machine Learning**: scikit-learn (Random Forest)
- **AI**: Google Gemini API
- **Data**: NASA Kepler Threshold Crossing Events

## License

This project is part of an ML internship program.

## Contact

- Email: poojakoppula4@gmail.com
- LinkedIn: [K. Pooja Reddy](https://www.linkedin.com/in/k-pooja-reddy-28p09)
- GitHub: [Repository](https://github.com/springboardmentor74280b-design/Habitability-of-Exoplanets/tree/K.POOJA-REDDY)
