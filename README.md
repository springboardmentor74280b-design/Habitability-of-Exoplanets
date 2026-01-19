# Exoplanet Habitability Prediction System

**Overview:**  
An AI-powered system that predicts and ranks the habitability of exoplanets using machine learning, coupled with a Flask-based web application for interactive exploration. Developed using VS Code and hosted on GitHub for version control.


## Project Description
The system analyzes planetary and stellar parameters to determine the potential habitability of exoplanets. It leverages real-world datasets, applies machine learning techniques, and presents results via an intuitive web interface.


## Key Features
- Machine learning–based habitability prediction  
- Habitability score calculation and planet ranking  
- Flask REST API backend  
- Interactive frontend using HTML, CSS, and JavaScript  
- Data visualizations with plots and charts  
- CSV-based dataset handling  

---

## Project Structure

Habitability-of-Exoplanets/
│
├── plots/ # Visualization images (heatmaps, charts)
├── static/ # Frontend assets (CSS, JS, images)
│
├── app.py # Flask backend
├── index.html # Web interface
│
├── training.py # Model training
├── habitability_prediction.py # Prediction logic
├── EXO.py # Data preprocessing and cleaning
├── dashboard.py # Visualization dashboard logic
│
├── model.pkl # Trained ML model
├── ranked_exoplanets.csv # Ranked exoplanets based on habitability
├── exoplanet_cleaned_final.csv # Cleaned dataset
├── phl_exoplanet_catalog_2019.csv # Raw dataset
│
├── README.md # Project documentation
└── License.txt # License information

yaml
Copy code

---

## Dataset
**Sources:**  
- NASA Exoplanet Archive  
- PHL Exoplanet Catalog  

**Features Used:**  
- Planet mass, radius, surface temperature, orbital period  
- Stellar mass, radius, temperature  

---

## Machine Learning Workflow
1. Data collection and preprocessing  
2. Feature engineering and normalization  
3. Model training  
4. Habitability prediction  
5. Exoplanet ranking  
6. Visualization and UI display  

---

## How to Run
**Step 1:** Clone the repository  
```bash
git clone <repository-url>
cd Habitability-of-Exoplanets
Step 2: Install dependencies

bash
Copy code
pip install -r requirements.txt
Step 3: Train the model (optional)

bash
Copy code
python training.py
Step 4: Run the application

bash
Copy code
python app.py
Step 5: Open in browser

Open index.html locally

Or visit the live deployment: Render App

API Endpoints
Endpoint	Method	Description
/predict	POST	Predict habitability
/rank	GET	Get ranked exoplanets
/health	GET	API status check

Technologies Used
Python, scikit-learn, pandas, numpy

Flask, HTML, CSS, JavaScript, Bootstrap

Matplotlib, Seaborn

Git & GitHub, VS Code

Applications
Identifying potentially habitable exoplanets

Astronomical data analysis

Academic ML projects

Full-stack data science demonstrations

Application Output
Habitability Status Pie Chart: Distribution of exoplanets by predicted habitability

Dashboard & Feature Visualizations: Influence of planetary features on habitability

Exoplanet Ranking Table: Ranked exoplanets by habitability score

Prediction Result: Displays habitability score and status for user-provided input

Deployment
The project is deployed on Render.

Live URL: https://habitability-of-exoplanets-2.onrender.com/

Video Demo: https://onedrive.live.com/?qt=allmyphotos&photosData=%2Fshare%2F7455FACDCC191830%21s7ba3f36466c141838e8686b421bd15d1%3Fithint%3Dvideo%26e%3DGx7JbM%26migratedtospo%3Dtrue&cid=7455FACDCC191830&id=7455FACDCC191830%21s7ba3f36466c141838e8686b421bd15d1&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3YvYy83NDU1ZmFjZGNjMTkxODMwL0lRQms4Nk43d1dhRFFZNkdoclFodlJYUkFaUUhGSDJZb3ItSVAxand4RzQzQ1dnP2U9R3g3SmJN&v=photos


Deployment Steps:

Push code to GitHub

Connect repository to Render

Install dependencies via requirements.txt

Start Flask app using Gunicorn

Verify live deployment

Author
Rushitha Konangi
B.Tech, Electronics & Communication Engineering
Infosys Springboard Program

License
Licensed under terms specified in License.txt.

Acknowledgments
Infosys Springboard Program

NASA Exoplanet Archive

PHL Exoplanet Catalog

Open-source Python & ML community


