Exoplanet Habitability Prediction System
Overview

An AI-powered system that predicts and ranks the habitability of exoplanets using machine learning, integrated with a Flask-based web application for interactive exploration. The project is developed using VS Code, version-controlled with GitHub, and deployed on Render for public access.

Project Description

The Exoplanet Habitability Prediction System analyzes planetary and stellar parameters to assess the potential habitability of exoplanets. It leverages real-world astronomical datasets, applies machine learning techniques for prediction and ranking, and presents results through an intuitive, interactive web interface with visual insights.

Key Features

Machine learning–based habitability prediction

Habitability score calculation and exoplanet ranking

Flask REST API backend

Interactive frontend using HTML, CSS, and JavaScript

Data visualizations with charts and plots

CSV-based dataset handling

Live web deployment for real-time access

Project Structure
Habitability-of-Exoplanets/
│
├── plots/                         # Visualization images (heatmaps, charts)
├── static/                        # Frontend assets (CSS, JS, images)
├── app.py                         # Flask backend
├── index.html                     # Web interface
├── training.py                    # Model training
├── habitability_prediction.py     # Prediction logic
├── EXO.py                         # Data preprocessing and cleaning
├── dashboard.py                   # Visualization dashboard logic
├── model.pkl                      # Trained ML model
├── ranked_exoplanets.csv          # Ranked exoplanets by habitability
├── exoplanet_cleaned_final.csv    # Cleaned dataset
├── phl_exoplanet_catalog_2019.csv # Raw dataset
├── README.md                      # Project documentation
└── License.txt                    # License information

Dataset
Data Sources

NASA Exoplanet Archive

PHL Exoplanet Catalog

Features Used

Planet mass

Planet radius

Surface temperature

Orbital period

Stellar mass

Stellar radius

Stellar temperature

Machine Learning Workflow

Data collection and preprocessing

Feature engineering and normalization

Model training using machine learning algorithms

Habitability prediction

Exoplanet ranking based on habitability score

Visualization and UI display

How to Run the Project
Step 1: Clone the Repository
git clone <repository-url>
cd Habitability-of-Exoplanets

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Train the Model (Optional)
python training.py

Step 4: Run the Application
python app.py

Step 5: Open in Browser

Open index.html locally
OR

Access the live deployed application

API Endpoints
Endpoint	Method	Description
/predict	POST	Predict habitability
/rank	GET	Get ranked exoplanets
/health	GET	API status check
Technologies Used

Programming & ML: Python, scikit-learn, pandas, numpy

Web Framework: Flask

Frontend: HTML, CSS, JavaScript, Bootstrap

Visualization: Matplotlib, Seaborn

Tools: Git, GitHub, VS Code

Applications

Identifying potentially habitable exoplanets

Astronomical data analysis

Academic machine learning projects

Full-stack data science demonstrations

Application Output

Habitability Status Pie Chart showing distribution of predicted habitability

Dashboard and feature visualizations highlighting influential planetary parameters

Exoplanet ranking table based on habitability score

Prediction result displaying habitability score and status for user input

Deployment
Live Deployment

Platform: Render

Live URL:
https://habitability-of-exoplanets-2.onrender.com/

Video Demo

https://onedrive.live.com/?qt=allmyphotos&photosData=%2Fshare%2F7455FACDCC191830%21s7ba3f36466c141838e8686b421bd15d1%3Fithint%3Dvideo%26e%3DGx7JbM

Deployment Steps

Push code to GitHub

Connect repository to Render

Install dependencies via requirements.txt

Start Flask app using Gunicorn

Verify live deployment

Author

Rushitha Konangi
B.Tech – Electronics & Communication Engineering
Infosys Springboard Program

License

Licensed under the terms specified in License.txt.

Acknowledgments

Infosys Springboard Program

NASA Exoplanet Archive

PHL Exoplanet Catalog

Open-source Python and Machine Learning community
