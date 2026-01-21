
# 🪐 ExoHab SaaS: AI-Powered Exoplanet Habitability Analyzer

### 📖 Project Overview
**ExoHab** is a full-stack Machine Learning application developed to analyze astronomical data directly from the **NASA Exoplanet Archive**. It utilizes advanced algorithms to predict whether an exoplanet is potentially habitable based on its physical and stellar parameters.

The core problem with current astronomical methods is the reliance on incomplete data. ExoHab bridges this gap using a custom **Physics Engine** that imputes missing values (like Mass and Luminosity) using standard astrophysical formulas (e.g., Kepler's Third Law, Stefan-Boltzmann Law).

Unlike standard "black-box" models, this project focuses on **Explainable AI (XAI)**. By integrating **SHAP (Shapley Additive Explanations)**, the system provides real-time transparency for every prediction, explaining *why* a planet is classified as habitable (e.g., "Planet Radius is 1.1 Earths" vs "Star Temperature is too high").

### 🚀 Key Features Implemented

* **⚛️ Physics Engine Integration**
    * **Intelligent Imputation:** Automatically calculates missing Orbital Periods and Semi-Major Axes using **Kepler’s Third Law**.
    * **Stellar Derivation:** Derives Star Luminosity and Mass from spectral types and temperatures using **Stefan-Boltzmann laws**.

* **🤖 Advanced Machine Learning Pipeline**
    * **Model:** XGBoost Classifier (Gradient Boosting) optimized for tabular data.
    * **Data Balancing:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to fix the inherent bias against rare habitable planets in the dataset.
    * **Validation:** Trained on the PHL (Planetary Habitability Laboratory) catalog for high-confidence ground truth.

* **🧠 Explainability (Glass Box AI)**
    * Integrated **SHAP** values to generate local force plots.
    * Users can click **"Why?"** on any result to see a breakdown of positive vs. negative feature impacts.

* **🌌 Interactive 3D Visualization**
    * A custom-built 3D Galaxy Map using **Plotly.js**.
    * Visualizes the "Goldilocks Zone" by plotting **Star Temperature vs. Planet Radius vs. Equilibrium Temperature** in a rotatable 3D space.

* **⚡ Bulk Data Processing**
    * **High-Performance:** Capable of processing raw NASA CSV files with 5,000+ entries in seconds.
    * **Stateless Architecture:** Designed as a SaaS tool where users can bring their own data (BYOD) without permanent server-side storage, ensuring privacy

### 🛠️ Tech Stack

| Category         | Technologies |
| :--------------- | :----------- |
| **Frontend**     | HTML5, CSS3, Bootstrap 5, JavaScript (Plotly.js) |
| **Backend**      | Python 3.10, Flask (REST API) |
| **Data Science** | Pandas, NumPy, Scikit-Learn, Imbalanced-Learn |
| **AI / ML**      | XGBoost, SHAP, Joblib |
| **Deployment**   | Render / Gunicorn |

### ⚙️ Installation & Setup
Follow these steps to run ExoHab locally:

**1. Clone the Repository**
```bash
git clone [https://github.com/springboardmentor74280b-design/Habitability-of-Exoplanets/tree/maneeswara_venkata_sai](https://github.com/springboardmentor74280b-design/Habitability-of-Exoplanets/tree/maneeswara_venkata_sai) 

cd exo_hab-ai

```

**2. Create a Virtual Environment**

```bash
# Windows:
python -m venv venv
venv\Scripts\activate

# Mac/Linux:
python3 -m venv venv
source venv/bin/activate

```

**3. Install Dependencies**

```bash
pip install -r requirements.txt

```

**4. Run the Application**

```bash
python app.py

```

*Access the dashboard at `http://127.0.0.1:5000*`

---

### 🔗 Quick Links

* **🎬 Video Demo:** [Watch on Google Drive](https://drive.google.com/drive/folders/1zaYkQRJkYoPbslNUh1DmqZ_rOHw94xdM?usp=drive_link)
* **🌐 Live Deployment:** [Visit ExoHab AI](https://exohabai.onrender.com/)

---

### 📂 Project Structure

```text
exo_hab-ai/
│
├── app.py                              # Main Flask Application (Entry Point)
├── dashboard_logic.py                  # PCA, t-SNE, and Plot Generation
├── exohab_model.joblib                 # Saved ML Model
├── explainability.py                   # SHAP Value Calculation Engine
├── LICENSE
├── model_utils.py                      # Kepler's Laws & Data Imputation Logic
├── phl_exoplanet_catalog.csv           # Dataset used to train the model
├── Procfile                            # Deployment Configuration (Render/Heroku)
├── Project learning document.docx
├── README.md
├── requirements.txt                    # Project Dependencies
├── train_model.py                      # ML Pipeline (XGBoost + SMOTE)
│
├── model_training/                     # Training Scripts & Experimental Data
│   ├── 4_models_comparison.py
│   ├── baseline_model.py
│   ├── Final_Habitable_Exoplanet_Report.xlsx
│   ├── final_pipeline.pkl
│   ├── final_test.py
│   ├── generate_ranking.py
│   ├── smote+xgboost.py
│   └── xgboost_type_comparison.py
│
├── plots/                              # Model Evaluation Visualizations
│   ├── 4_model_comparison_plot.png
│   ├── nasa_confusion_matrix.png
│   ├── nasa_roc_curve.png
│   └── projection_plots.png
│
└── templates/                          # Frontend UI Templates
    ├── analyze.html
    ├── base.html
    ├── dashboard.html
    └── home.html

```

---

### 👨‍💻 Contributors

* **Avula Maneeswara Venkata Sai** 


---

### 🔮 Future Scope

* **Real-time NASA API Sync:** Automating the data fetch process to get the latest daily discoveries.
* **Procedural Planet Generation:** Using Three.js to render planet textures based on atmospheric composition.
* **User Accounts:** Implementing Google OAuth for saving personalized analysis reports.

```

```