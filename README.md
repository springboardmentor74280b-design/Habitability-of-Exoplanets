# 🪐 ExoHab SaaS: AI-Powered Exoplanet Habitability Analyzer

## 📖 Project Overview
**ExoHab** is a full-stack Machine Learning application developed to analyze astronomical data from the NASA Exoplanet Archive. It utilizes advanced algorithms to predict whether an exoplanet is potentially habitable based on its physical and stellar parameters.

Unlike standard black-box models, this project focuses on **Explainable AI (XAI)**, providing real-time reasoning for every prediction using SHAP values.

## 🚀 Key Features Implemented
1.  **Physics Engine Integration**:
    * Implements Kepler’s Third Law to calculate missing Orbital Periods/Distances.
    * Derives Stellar Luminosity and Mass using Stefan-Boltzmann laws when data is missing.
2.  **Advanced Machine Learning**:
    * **Model**: XGBoost Classifier (Gradient Boosting).
    * **Handling Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to fix the bias against rare habitable planets.
3.  **Explainability (Glass Box AI)**:
    * Integrated **SHAP (Shapley Additive Explanations)**.
    * Users can click "Why?" on any prediction to see exactly which factors (e.g., Star Temp, Planet Radius) influenced the decision.
4.  **Interactive 3D Visualization**:
    * A custom-built 3D Galaxy Map using Plotly.js.
    * Visualizes the "Goldilocks Zone" by plotting Star Temperature vs. Planet Radius vs. Equilibrium Temperature.
5.  **Bulk Data Processing**:
    * Allows users to upload raw NASA CSV files.
    * Processes 4,000+ planets instantly with a "Performance Mode" toggle for visualization.

## 🛠️ Tech Stack
* **Frontend**: HTML5, Bootstrap 5, JavaScript (Plotly.js).
* **Backend**: Python, Flask.
* **Data Science**: Pandas, NumPy, Scikit-Learn, Imbalanced-Learn.
* **AI/ML**: XGBoost, SHAP.

VIDEO DEMO LINK : [text](https://drive.google.com/drive/folders/1zaYkQRJkYoPbslNUh1DmqZ_rOHw94xdM?usp=drive_link)
DEPLOYED EXOHABAI LINK : [text](https://exohabai.onrender.com/)

## 📂 Project Structure
```text
==============================
exo_hab-ai/
│   ├── app.py                              # Main Flask Application (Entry Point)
│   ├── dashboard_logic.py                  # PCA, t-SNE, and Plot Generation
│   ├── exohab_model.joblib
│   ├── explainability.py                   # SHAP Value Calculation Engine
│   ├── LICENSE
│   ├── model_utils.py                      # Kepler's Laws & Data Imputation Logic
│   ├── phl_exoplanet_catalog.csv           # Dataset used tto train the model
│   ├── Procfile                            # Deployment Configuration
│   ├── Project learning document.docx
│   ├── README.md
│   ├── requirements.txtq                   # Project Dependencies
│   ├── train_model.py                      # ML Pipeline (XGBoost + SMOTE)
│   ├── model_training/                     # Contains training and testing data of model
│   │   ├── 4_models_comparison.py
│   │   ├── baseline_model.py
│   │   ├── Final_Habitable_Exoplanet_Report.xlsx
│   │   ├── final_pipeline.pkl
│   │   ├── final_test.py
│   │   ├── generate_ranking.py
│   │   ├── imputer.pkl
│   │   ├── logistic_regression.py
│   │   ├── phl_exoplanet_catalog.csv
│   │   ├── projection_plots.py
│   │   ├── PSCompPars_2025.12.24_23.44.10.csv
│   │   ├── ranked_planets_leaderboard.csv
│   │   ├── smote+rf.py
│   │   ├── smote+xgboost.py
│   │   ├── smote+xgboost_model_plots.py
│   │   ├── smote+xgboost_pipeline.py
│   │   ├── test2.py
│   │   ├── test3.py
│   │   ├── testing.py
│   │   ├── weighted_svm.py
│   │   ├── weighted_xgboost.py
│   │   ├── xgboost_pipeline_generator.py
│   │   ├── xgboost_type_comparison.py
│   ├── plots/                               # Contains plots of the models 
│   │   ├── 4_model_comparison_plot.png
│   │   ├── baseline_model_plots.png
│   │   ├── logistic regression.png
│   │   ├── nasa_confusion_matrix.png
│   │   ├── nasa_roc_curve.png
│   │   ├── physics_confusion_matrix.png
│   │   ├── projection_plots.png
│   │   ├── smote+random forest.png
│   │   ├── smote+xgboost_1.png
│   │   ├── svm_weighted.png
│   │   ├── s_xg_full.png
│   │   ├── weighted_xgboost.png
│   │   ├── xgboost_type_comparison.png
│   │   ├── xgboost_w.png
│   ├── templates/                          # Contains main UI templates
│   │   ├── analyze.html
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── home.html
==============================