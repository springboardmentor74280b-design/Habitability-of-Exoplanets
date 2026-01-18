
# ExoHabitAI – Habitability of Exoplanets 🌍🪐

## Project Overview
ExoHabitAI is a machine learning–powered web application that predicts the habitability of exoplanets based on astrophysical parameters. The project integrates data preprocessing, ML modeling, backend APIs, and frontend visualization into a complete end-to-end system.

## Features
- Predict exoplanet habitability score and label
- Rank top potentially habitable exoplanets
- Interactive data visualizations using Plotly
- REST API built with Flask

## Tech Stack
### Frontend
- HTML
- CSS
- JavaScript
- Axios
- Plotly

### Backend
- Python
- Flask
- Flask-SQLAlchemy
- REST APIs

### Machine Learning
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Pandas
- Joblib

### Database
- SQLite

## Project Structure
```
ExoHabitAI/
│
├── Habitability-of-Exoplanets/
│   │
│   ├── artifacts/
│   │   ├── final_model.pkl
│   │   └── ranked_exoplanets.csv
│   │
│   ├── backend/
│   │   ├── __pycache__/
│   │   ├── instance/
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── exoplanets.py
│   │   │   ├── predict.py
│   │   │   └── ranking.py
│   │   │
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── feature_ranges.py
│   │   ├── inspect_model.py
│   │   ├── models.py
│   │   └── requirements.txt
│   │
│   ├── data/
│   │   ├── exoplanets_raw.csv
│   │   ├── exoplanets_cleaned.csv
│   │   └── exoplanets_validated.csv
│   │
│   ├── frontend/
│   │   ├── css/
│   │   │   └── styles.css
│   │   │
│   │   ├── js/
│   │   │   ├── app.js
│   │   │   ├── ranking.js
│   │   │   └── visualization.js
│   │   │
│   │   ├── index.html
│   │   ├── ranking.html
│   │   └── visualization.html
│   │
│   ├── reports/
│   │   ├── figures/
│   │   │── radius_vs_score.html
│   │   │── temperature_vs_score.html
│   │   ├── top_20_habitable_exoplanets.csv
│   │   └── top_20_habitable_exoplanets.xlsx
│   │
│   ├── scripts/
│   │   ├── clean_data.py
│   │   ├── collect_data.py
│   │   ├── correlation_heatmap.py
│   │   ├── eda_pca_tsne.py
│   │   ├── export_top_candidates.py
│   │   ├── feature_importance.py
│   │   ├── habitability_distribution.py
│   │   ├── ml_baseline.py
│   │   ├── ml_smote_full.py
│   │   ├── plotly_visuals.py
│   │   ├── rank_exoplanets.py
│   │── bivariate_analysis.py
│   │── bivariate_summary.csv
│   │── correlation_matrix.csv
│   ├── LICENSE
│   ├── Procfile
│   ├── Project learning document.docx
│   ├── README.md
│   ├── requirements.txt
│   └── venv/

```

## How to Run Locally
```bash
python -m backend.app
```

Then open:
```
http://127.0.0.1:5000
```



## Author
Developed as part of an academic internship / project module.
