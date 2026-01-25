# 🌍 Exoplanet Habitability Prediction using XGBoost

## 📌 Project Overview
This project predicts the **habitability of exoplanets** using machine learning techniques by analyzing planetary and stellar characteristics. Since the original dataset does not contain a predefined *habitability* label, a **custom habitability target variable was engineered** based on scientifically relevant planetary features.

The project was developed as part of the **Infosys Springboard Internship Program**, under the guidance of **Mentor: Bhanu Prakash Sir**.

---

## 🎯 Problem Statement
To build an intelligent system that can classify exoplanets as **habitable or non-habitable** based on physical and orbital parameters, and deploy the model as a real-time web application.

---

## 🧠 Target Variable Engineering (Habitability Creation)
The dataset did **not contain a `habitability` column**. Therefore, a target variable was created using scientifically motivated thresholds derived from planetary and stellar features such as:

- Equilibrium Temperature
- Stellar Flux
- Orbital Period
- Planet Mass and Radius
- Stellar Temperature

Planets satisfying Earth-like ranges were labeled as:
- `1` → Habitable  
- `0` → Non-Habitable  

This feature-engineered target enabled supervised learning and improved model interpretability.

---

## 🧠 Machine Learning Model
- **Final Model:** XGBoost Classifier  
- **Why XGBoost?**
  - Handles complex non-linear relationships
  - Robust to noise and outliers
  - Works well with tabular astronomical data
  - Provides probability-based predictions

---

## 🗂️ Dataset Description
The dataset contains planetary and stellar attributes including:
- Orbital Period
- Planet Mass and Radius
- Orbital Eccentricity
- Equilibrium Temperature
- Stellar Flux
- Stellar Temperature
- Stellar Metallicity
- Stellar Radius
- Stellar Mass
- Stellar Surface Gravity
- Stellar Distance

---

## 🛠️ Data Preprocessing
The following preprocessing steps were applied:

### Missing Values
- Median imputation for numerical features
- Chosen due to robustness against skewed astronomical data

### Outlier Detection
- Interquartile Range (IQR) method
- Suitable for non-normal distributions

### Feature Scaling
- StandardScaler applied

### Encoding
- One-Hot Encoding used for categorical features

---

## ⚖️ Handling Class Imbalance using SMOTE
The engineered habitability target was **highly imbalanced**, with fewer habitable planets compared to non-habitable ones.

To address this issue:
- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied
- It generates synthetic samples of the minority class
- Prevents model bias toward majority class
- Improves recall and F1-score for habitable planets

This step significantly improved model generalization.

---

## ⚙️ Model Training & Evaluation
- Train-test split applied
- SMOTE applied on training data
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

XGBoost outperformed baseline models and was selected as the final model.

---

## 🌐 Deployment
The trained model was deployed as a **Flask-based web application** and hosted on **Render**.

### 🔗 Live Application
👉 https://habitability-of-exoplanets-62ko.onrender.com

---

## 🧪 Sample Prediction (Earth-like Case)

### Input
- Orbital Period: 365 days  
- Planet Mass: 1.0  
- Planet Radius: 1.0  
- Orbital Eccentricity: 0.0167  
- Equilibrium Temperature: 288 K  
- Stellar Flux: 1.0  
- Stellar Temperature: 5777 K  
- Stellar Metallicity: 0.0  
- Stellar Radius: 1.0  
- Stellar Mass: 1.0  
- Stellar Surface Gravity: 4.44  
- Stellar Distance: 10 parsecs  

### Output
Habitability: 1
Probability: 0.9977
Earth-like conditions detected. You could survive here.

## 📁 Project Structure:
```bash
habitability/
│
├── api/
│ ├── init.py
│ ├── app.py # Flask entry point
│ ├── models.py # Model loading & inference
│ ├── routes.py # API routes
│ │
│ ├── static/
│ │ ├── hab.png
│ │ ├── feature_importance.png
│ │ ├── habitability_distribution.png
│ │ ├── correlation_heatmap.png
│ │ ├── top10_planets.png
│ │
│ └── templates/
│ └── index.html # Frontend UI
│
├── dashboard/
│ ├── dashboard.py # Visualization dashboard
│ └── plots.py # Plot generation
│
├── models/
│ ├── preprocessing.py # Cleaning & scaling
│ ├── feature_engineering.py# Habitability creation logic
│ ├── prepare_ml_data.py # Train-test preparation
│ ├── train_model.py # Model training
│ ├── xgboost_model.joblib # Saved model
│ └── scaler.joblib # Saved scaler
│
├── data/
│ ├── exoplanet_habitable.csv
│ ├── selected_features.csv
│ └── top_exoplanets.csv
│
├── venv/ # Virtual environment
│
└── requirements.txt
```


---

## 🚀 Technologies Used
- Python
- XGBoost
- Scikit-learn
- Pandas, NumPy
- Flask
- HTML, CSS,JS
- REST APIs
- Render (Cloud Deployment)

---

## 🔧 Installation and Setup

Follow the steps below to set up and run the project locally

---

### 1️⃣ Clone the Repository
### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
py models/preprocessing.py
py models/feature_engineering.py
py models/prepare_ml_data.py
py models/train_model.py
py dashboard/dashboard.py
py dashboard/plots.py
py -m api.app
```
### Render App Link:
https://habitability-of-exoplanets-62ko.onrender.com/#



## 👨‍🏫 Internship Details
- **Program:** Infosys Springboard Internship  
- **Mentor:** Bhanu Prakash Sir  
- **Domain:** Machine Learning / Data Science  

---

## 📈 Key Outcomes
- Created a target variable from raw astronomical features
- Handled severe class imbalance using SMOTE
- Built an end-to-end ML pipeline
- Successfully deployed a real-time ML system
- Gained practical experience in cloud deployment

---

## 🔮 Future Enhancements
- Integration with live NASA Exoplanet Archive
- Improved visualization dashboard
- Mobile-responsive UI

---
## Contributor:
Middepaka Megha Chandrika

---

## 🙏 Acknowledgements
I sincerely thank **Infosys Springboard** and my mentor **Bhanu Prakash Sir** for their continuous guidance and support throughout this internship.

---

⭐ If you find this project useful, feel free to star the repository!
