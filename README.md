# 🪐 Exoplanet Habitability Prediction System

An AI-powered system that predicts and ranks the habitability of exoplanets using machine learning and a Flask-based web application.

The project is developed using **VS Code** and hosted on **GitHub** for version control.

---

## Project Overview

The **Exoplanet Habitability Prediction System** analyzes planetary and stellar parameters to determine whether an exoplanet can potentially support life.

It uses real-world datasets, applies machine learning techniques, and displays results through a simple and interactive web interface.

---

## 🌟 Features

- Machine learning–based habitability prediction  
- Habitability score calculation and planet ranking  
- Flask REST API backend  
- Interactive frontend using HTML, CSS, and JavaScript  
- Data visualization using plots and charts  
- CSV-based dataset handling  

---

## 📂 Project Structure

Habitability-of-Exoplanets/
├── plots/ # Visualization images
├── static/ # Frontend files
├── app.py # Flask backend
├── index.html # Web UI
├── training.py # Model training
├── habitability_prediction.py # Prediction logic
├── EXO.py # Data processing
├── dashboard.py # Visualizations
├── model.pkl # Trained model
├── ranked_exoplanets.csv # Ranked output
├── exoplanet_cleaned_final.csv # Cleaned dataset
├── phl_exoplanet_catalog_2019.csv # Raw dataset
├── README.md # Documentation
└── License.txt # License


## 📊 Dataset

### Sources
- NASA Exoplanet Archive  
- PHL Exoplanet Catalog  

### Features Used
- Planet mass  
- Planet radius  
- Surface temperature  
- Orbital period  
- Stellar mass  
- Stellar radius  
- Stellar temperature  

---

## 🧠 Machine Learning Workflow

1. Data collection  
2. Data cleaning and preprocessing  
3. Feature engineering and normalization  
4. Model training  
5. Habitability prediction  
6. Exoplanet ranking  
7. Visualization and UI display  

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Habitability-of-Exoplanets
Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Train the Model (Optional)
python training.py
Step 4: Run the Application
python app.py
Step 5: Open in Browser
Open index.html
OR

Visit:
http://127.0.0.1:5000
📡 API Endpoints
Endpoint	Method	Description
/predict	POST	Predict habitability
/rank	GET	Get ranked exoplanets
/health	GET	API status check

🛠️ Technologies Used
Python
scikit-learn, pandas, numpy
Flask
HTML, CSS, JavaScript, Bootstrap
Matplotlib, Seaborn
Git & GitHub
VS Code
🎯 Applications
Identifying potentially habitable exoplanets
Astronomical data analysis
Machine learning academic projects
Full-stack data science demonstration

Application Output Screenshots
🪐 Habitability Status Pie Chart

This chart shows the distribution of exoplanets based on predicted habitability levels (High, Medium, Low).

📊 Dashboard & Feature Visualizations

This dashboard visualizes important planetary features and their influence on habitability prediction.

📋 Exoplanet Ranking Table

Exoplanets are ranked based on their predicted habitability scores, helping identify the most promising candidates.

🔮 Habitability Prediction Result

This screen displays the habitability score and status for a user-provided exoplanet input.

## 🚀 Deployment

The project is deployed on **Render**.

🔗 Live URL:
https://habitability-of-exoplanets-2.onrender.com
Video Demo: https://onedrive.live.com/?qt=allmyphotos&photosData=%2Fshare%2F7455FACDCC191830%21s7ba3f36466c141838e8686b421bd15d1%3Fithint%3Dvideo%26e%3DGx7JbM%26migratedtospo%3Dtrue&cid=7455FACDCC191830&id=7455FACDCC191830%21s7ba3f36466c141838e8686b421bd15d1&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3YvYy83NDU1ZmFjZGNjMTkxODMwL0lRQms4Nk43d1dhRFFZNkdoclFodlJYUkFaUUhGSDJZb3ItSVAxand4RzQzQ1dnP2U9R3g3SmJN&v=photos


### Deployment Steps
1. Pushed code to GitHub
2. Connected GitHub repo to Render
3. Installed dependencies using `requirements.txt`
4. Started Flask app using Gunicorn
5. Verified live deployment


👩‍💻 Author
Rushitha Konangi
B.Tech Final Year Student
Infosys Springboard Program

📜 License
This project is licensed under the terms specified in License.txt.
🙏 Acknowledgments
Infosys Springboard for the learning opportunity
NASA Exoplanet Archive
PHL Exoplanet Catalog
Open-source Python and ML community
