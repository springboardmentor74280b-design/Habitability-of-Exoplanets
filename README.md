# Exoplanet Habitability Prediction System

A machine learning-based system for predicting the habitability of exoplanets using advanced classification algorithms and data balancing techniques. This project was developed as part of the **Infosys Springboard Program**.

## Project Overview

The Exoplanet Habitability Prediction System analyzes astronomical data to classify exoplanets based on their potential habitability. The system employs multiple machine learning models, handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique), and provides predictions through a modern web interface.

### Key Features

- **Multi-Model Classification**: Implements Logistic Regression, K-Nearest Neighbors (KNN), and XGBoost algorithms
- **Class Imbalance Handling**: SMOTE technique for balanced training data
- **Comprehensive EDA**: Univariate and bivariate statistical analysis with visualizations
- **Dimensionality Reduction**: PCA and t-SNE for feature analysis
- **RESTful API**: FastAPI backend for model predictions
- **Interactive Frontend**: React-based user interface
- **Data Pipeline**: Complete ML pipeline from data cleaning to model deployment

## Technology Stack

### Backend
- **FastAPI**: High-performance web framework for building APIs
- **Scikit-Learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **Imbalanced-Learn**: SMOTE implementation for handling class imbalance
- **NumPy**: Numerical computing
- **Joblib**: Model serialization
- **Uvicorn**: ASGI server

### Frontend
- **React**: JavaScript library for building user interfaces
- **Modern UI Components**: Interactive prediction interface

### Machine Learning
- **Classification Models**: Logistic Regression, KNN, XGBoost
- **Data Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Engineering**: Scaling, PCA, t-SNE
- **Model Evaluation**: Confusion matrices, performance metrics

## Project Structure

![Project Structure](image_8a62f4.png)

```
Exoplanet-Habitability-Prediction/
│
├── 01_Descriptive_statistics/
│   ├── bivariate_analysis.py
│   └── univari_analysis.py
│
├── Backend/
│   ├── app.py                              # FastAPI application
│   ├── database.py                         # Database operations
│   ├── predict.py                          # Prediction logic
│   ├── model_loader.py                     # Model loading utilities
│   ├── feature_mapper.py                   # Feature mapping
│   ├── schemas.py                          # Pydantic schemas
│   ├── security.py                         # Security configurations
│   ├── requirements.txt                    # Python dependencies
│   ├── feature_order.json                  # Feature ordering configuration
│   ├── exoplanet_habitability_pipeline.pkl # Trained pipeline
│   └── saved_models/                       # Serialized models
│
├── frontend/                               # React application
│
├── eda/                                    # Exploratory data analysis outputs
├── EDA_plots/                              # Visualization outputs
├── outputs/                                # Model outputs
├── results/                                # Experiment results
├── saved_models/                           # Trained models
│
├── 02_Checking._Missing_value.py          # Missing value analysis
├── 03_Data_Cleaning.py                    # Data preprocessing
├── 04_Splitting_of_dataset.py             # Train-test split
├── 05_scaling.py                          # Feature scaling
├── 5.1_Visualize_scaling.py               # Scaling visualization
├── 06_EDA_exoplanet.py                    # Exploratory data analysis
├── 07_Baseline_models_before_smote.py     # Initial model training
├── 08_SMOTE_Traning.py                    # SMOTE application
├── 09_smote_visualizations.py             # SMOTE visualization
├── 10_baseline_models_after_smote.py      # Post-SMOTE model training
├── 11_PCA_t-SNE.py                        # Dimensionality reduction
├── 12_ML_pipeline_3models.py              # Complete ML pipeline
├── Data_Loader.py                         # Data loading utilities
├── visualize_confusion_matrices.py        # Model evaluation visualization
│
├── exoplanet_habitable_pres_dataset.csv   # Original dataset
├── cleaned_exoplanet_dataset.csv          # Preprocessed dataset
├── train_smote_exoplanet.csv              # SMOTE-balanced training data
│
└── README.md                              # Project documentation
```

## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- pip (Python package manager)
- npm or yarn (Node package manager)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Exoplanet-Habitability-Prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   cd Backend
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`
   
   API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm start
   # or
   yarn start
   ```

   The application will open at `http://localhost:3000`

## Usage

### Running the ML Pipeline

Execute the scripts in numerical order to reproduce the complete analysis:

```bash
# 1. Descriptive Statistics
python 01_Descriptive_statistics/univari_analysis.py
python 01_Descriptive_statistics/bivariate_analysis.py

# 2. Data Preprocessing
python 02_Checking._Missing_value.py
python 03_Data_Cleaning.py
python 04_Splitting_of_dataset.py

# 3. Feature Engineering
python 05_scaling.py
python 5.1_Visualize_scaling.py

# 4. Exploratory Data Analysis
python 06_EDA_exoplanet.py

# 5. Model Training (Before SMOTE)
python 07_Baseline_models_before_smote.py

# 6. SMOTE Application
python 08_SMOTE_Traning.py
python 09_smote_visualizations.py

# 7. Model Training (After SMOTE)
python 10_baseline_models_after_smote.py

# 8. Dimensionality Reduction
python 11_PCA_t-SNE.py

# 9. Final Pipeline
python 12_ML_pipeline_3models.py
```

### Making Predictions via API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": value1,
    "feature2": value2,
    ...
  }'
```

## Model Performance

### Model Performance Graphs
<!-- Insert your model performance comparison graphs here -->
![Model Performance](path/to/your/performance_graph.png)

### SMOTE Visualizations
<!-- Insert your SMOTE before/after visualizations here -->
![SMOTE Visualization](path/to/your/smote_visualization.png)

### Confusion Matrices

The system generates confusion matrices for each model:
- `Logistic_Regression_cm.png`
- `KNN_cm.png`
- Additional model confusion matrices in the `outputs/` directory

## Dataset

The project uses the **Exoplanet Habitability Dataset** containing astronomical measurements and features relevant to planetary habitability assessment.

**Dataset Files:**
- `exoplanet_habitable_pres_dataset.csv` - Original dataset
- `cleaned_exoplanet_dataset.csv` - Preprocessed dataset
- `train_smote_exoplanet.csv` - SMOTE-balanced training data

## API Endpoints

### Health Check
```
GET /
```

### Prediction
```
POST /predict
```
Request body: JSON object with exoplanet features

### API Documentation
```
GET /docs
```
Interactive Swagger UI documentation

## Contributors

- **Abirami** - Project Developer

## License

This project is part of the **Infosys Springboard Program**. All rights reserved.

## Acknowledgments

- Infosys Springboard for project guidance and support
- The exoplanet research community for dataset availability
- Open-source contributors of the libraries used in this project

---

**Note**: This project is developed for educational purposes as part of the Infosys Springboard Program.
