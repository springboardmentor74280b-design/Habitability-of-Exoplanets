# ExoScan - Deep Space Exoplanet Classifier

A high-end, immersive React web application for predicting exoplanet habitability using a Class-weighted SVM machine learning model.

## Features

- 🚀 **Immersive UI**: NASA Modern meets Cyberpunk aesthetic
- 🌌 **Real-time Visualization**: 3D planet preview that scales with input
- 📊 **Interactive Scanner**: Sleek form with sliders and numerical inputs
- 📈 **Prediction Results**: Habitability score gauge with confidence metrics
- ⚡ **Smooth Animations**: Framer Motion powered transitions

## Tech Stack

- **React 18** with Vite
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **Axios** for API communication

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Backend Setup

Ensure the FastAPI backend is running on `http://127.0.0.1:8000`:

```bash
cd Backend
uvicorn app:app --reload
```

## API Endpoint

The frontend communicates with:
- **Endpoint**: `POST http://127.0.0.1:8000/api/predict`
- **Request Format**:
```json
{
  "features": {
    "planet_radius": 1.0,
    "planet_mass": 1.0,
    "orbital_period": 365.0,
    "stellar_temperature": 5778.0,
    "stellar_luminosity": 1.0,
    "planet_density": 5.5,
    "semi_major_axis": 1.0
  }
}
```

## Color Palette

- **Deep Space Blue**: `#0B0D17`
- **Neon Cyan**: `#00F2FF`
- **Galactic Purple**: `#7000FF`
- **Stellar Gold**: `#FFD700`

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Hero.jsx              # Landing page hero section
│   │   ├── ScannerForm.jsx       # Input form with sliders
│   │   ├── PlanetVisualization.jsx  # 3D planet preview
│   │   └── PredictionResults.jsx    # Results display
│   ├── api/
│   │   └── predict.js            # API integration
│   ├── App.jsx                   # Main application component
│   ├── main.jsx                  # Entry point
│   └── index.css                 # Global styles
├── index.html
├── package.json
├── tailwind.config.js
└── vite.config.js
```

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Features in Detail

### Hero Landing
- Glassmorphic card design
- "Scan the Stars" call-to-action
- Smooth scroll to input section

### Input Dashboard
- 7 key exoplanet features
- Dual input: sliders + numerical inputs
- Real-time planet visualization updates
- Glow effects on interactive elements

### Prediction Results
- Circular habitability score gauge
- Classification with icons
- Technical specifications footer
- MCC (0.82) and ROC-AUC metrics

## Model Information

- **Model Type**: Class-weighted SVM (3-class)
- **MCC Score**: 0.82
- **Classification**: Non-Habitable (0), Habitable (1), Likely Habitable (2)
- **Data Source**: NASA Exoplanet Archive

## License

MIT
