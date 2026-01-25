# 🌍 Exoplanet Habitability Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting the **habitability of exoplanets** using machine learning techniques. By analyzing planetary and stellar parameters, the system classifies planets as **habitable or non-habitable** with high accuracy.  

The project was developed as part of the **Infosys Springboard Internship Program**, under the guidance of **Mentor: Bhanu Prakash Sir**.

---

## 🎯 Objectives
- To analyze exoplanet and stellar data
- To preprocess astronomical datasets effectively
- To build a robust machine learning model for habitability prediction
- To deploy the trained model as a web application on the cloud
- To provide real-time predictions with probability scores

---

## 🧠 Machine Learning Model
- **Final Model:** XGBoost Classifier  
- **Reason for Selection:**
  - Handles non-linear relationships effectively
  - Robust to noise and outliers
  - High performance on structured tabular data
  - Supports probability-based predictions

---

## 🗂️ Dataset Description
The dataset contains planetary and stellar attributes such as:
- Orbital Period
- Planet Mass and Radius
- Orbital Eccentricity
- Equilibrium Temperature
- Stellar Flux
- Stellar Temperature
- Stellar Metallicity
- Stellar Radius and Mass
- Stellar Surface Gravity
- Stellar Distance

---

## 🛠️ Data Preprocessing
The following preprocessing steps were applied:

- **Missing Value Handling**
  - Median imputation used for numerical features
  - Chosen due to robustness against skewed data and outliers

- **Outlier Detection**
  - Interquartile Range (IQR) method
  - Suitable for non-normal astronomical distributions

- **Feature Scaling**
  - StandardScaler applied to normalize numerical features

- **Encoding**
  - One-Hot Encoding used for categorical variables

---

## ⚙️ Model Training & Evaluation
- Train-test split applied
- Class imbalance handled using SMOTE
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

The XGBoost model demonstrated superior performance compared to baseline models.

---

## 🌐 Deployment
The trained model was deployed on **Render**.

### 🔗 Live Application Link
👉 https://habitability-of-exoplanets-62ko.onrender.com

---

## 🧪 Sample Prediction (Earth-like Case)

### Input Values
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
-Habitability: 1
-Probability: 0.9977
-Message: Earth-like conditions detected. You could survive here.
## 🚀 Technologies Used
- Python
- XGBoost
- Scikit-learn
- Pandas, NumPy
- Flask
- HTML, CSS
- Render (Cloud Deployment)

---

## 👨‍🏫 Internship Details
- **Program:** Infosys Springboard Internship  
- **Mentor:** Bhanu Prakash Sir  
- **Domain:** Machine Learning / Data Science  

---

## 📈 Key Outcomes
- Built an end-to-end ML pipeline
- Achieved high prediction accuracy
- Successfully deployed a real-time ML application
- Gained hands-on experience in cloud deployment

---

## 🙌 Acknowledgements
I would like to express my sincere gratitude to **Infosys Springboard** and my mentor **Bhanu Prakash Sir** for their guidance and support throughout this internship project.

---

⭐ If you find this project useful, feel free to star the repository!
