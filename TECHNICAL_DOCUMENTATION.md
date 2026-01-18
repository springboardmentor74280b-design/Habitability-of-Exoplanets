# Technical Documentation: Exoplanet Habitability Prediction System

## 1. Introduction

### Project Overview
The Exoplanet Habitability Prediction System is a machine learning-based application designed to classify exoplanets based on their potential habitability. The system processes astronomical data and employs multiple classification algorithms to predict whether an exoplanet falls into one of three habitability categories: Non-Habitable, Habitable, or Likely Habitable.

### Problem Statement
With the discovery of thousands of exoplanets, determining their potential for supporting life has become a critical challenge in astronomy. Manual analysis of planetary characteristics is time-consuming and subjective. This system addresses the need for automated, data-driven habitability assessment using machine learning techniques.

### Objective of the System
- Automate the classification of exoplanets based on habitability potential
- Provide confidence scores for predictions to assist astronomers in decision-making
- Handle class imbalance in habitability data using advanced sampling techniques
- Deliver predictions through a user-friendly web interface and RESTful API

## 2. System Architecture

### High-level Architecture Explanation
The system follows a three-tier architecture:

1. **Data Layer**: Raw astronomical datasets, preprocessed data, and trained model artifacts
2. **Processing Layer**: Machine learning pipeline including data preprocessing, feature engineering, model training, and prediction logic
3. **Presentation Layer**: FastAPI backend serving predictions and React frontend for user interaction

### Data Flow Overview
```
Raw Dataset → Data Cleaning → Feature Engineering → SMOTE Balancing → Model Training → Model Persistence → API Deployment → Frontend Interface
```

### Model Training vs Inference Workflow

**Training Workflow:**
1. Load raw exoplanet dataset
2. Perform missing value analysis and data cleaning
3. Split dataset into training and testing sets
4. Apply feature scaling and engineering
5. Handle class imbalance using SMOTE
6. Train multiple classification models
7. Evaluate and compare model performance
8. Serialize best-performing model pipeline

**Inference Workflow:**
1. Receive feature data via API endpoint
2. Load pre-trained model pipeline
3. Apply same preprocessing transformations
4. Generate prediction and confidence score
5. Return structured response to client

## 3. Technology Stack

### Programming Languages
- **Python 3.8+**: Primary language for machine learning pipeline and backend API
- **JavaScript**: Frontend development with React

### Frameworks & Libraries
- **FastAPI 0.104.1**: High-performance web framework for API development
- **React**: Frontend user interface framework
- **Pydantic 2.5.0**: Data validation and serialization
- **Uvicorn 0.24.0**: ASGI server for FastAPI deployment

### ML Tools Used
- **Scikit-Learn 1.3.2**: Core machine learning algorithms and preprocessing
- **XGBoost 2.0.3**: Gradient boosting classifier
- **Imbalanced-Learn 0.11.0**: SMOTE implementation for handling class imbalance
- **NumPy 1.26+**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Joblib 1.3.2**: Model serialization and persistence

### Deployment Tools
- **CORS Middleware**: Cross-origin resource sharing for frontend-backend communication
- **Virtual Environment**: Isolated Python environment management

## 4. Folder Structure Explanation

### Root Level Files
- **Data_Loader.py**: Utility functions for loading and splitting datasets
- **visualize_confusion_matrices.py**: Model evaluation visualization utilities
- **exoplanet_habitable_pres_dataset.csv**: Original raw dataset
- **cleaned_exoplanet_dataset.csv**: Preprocessed dataset after cleaning
- **train_smote_exoplanet.csv**: SMOTE-balanced training dataset

### 01_Descriptive_statistics/
Contains statistical analysis scripts for initial data exploration:
- **bivariate_analysis.py**: Correlation analysis between features
- **univari_analysis.py**: Individual feature distribution analysis

### Backend/
FastAPI application and model serving infrastructure:
- **app.py**: Main FastAPI application with CORS configuration
- **predict.py**: Prediction endpoint logic and feature mapping
- **model_loader.py**: Model loading utilities with caching
- **schemas.py**: Pydantic data models for request/response validation
- **database.py**: Database operations (if applicable)
- **security.py**: Security configurations
- **feature_mapper.py**: Feature name mapping utilities
- **requirements.txt**: Python dependencies specification
- **feature_order.json**: Feature ordering configuration for model input
- **exoplanet_habitability_pipeline.pkl**: Serialized trained model pipeline
- **saved_models/**: Directory for additional model artifacts

### frontend/
React application for user interface (structure not detailed in current project state)

### Data Processing Scripts (Numbered Sequence)
- **02_Checking._Missing_value.py**: Missing value analysis
- **03_Data_Cleaning.py**: Data preprocessing and cleaning
- **04_Splitting_of_dataset.py**: Train-test dataset splitting
- **05_scaling.py**: Feature scaling implementation
- **5.1_Visualize_scaling.py**: Scaling transformation visualization
- **06_EDA_exoplanet.py**: Comprehensive exploratory data analysis
- **07_Baseline_models_before_smote.py**: Initial model training without balancing
- **08_SMOTE_Traning.py**: SMOTE application and balanced model training
- **09_smote_visualizations.py**: SMOTE transformation visualization
- **10_baseline_models_after_smote.py**: Model training with SMOTE-balanced data
- **11_PCA_t-SNE.py**: Dimensionality reduction analysis
- **12_ML_pipeline_3models.py**: Complete ML pipeline with three classification models

### Output Directories
- **eda/**: Exploratory data analysis outputs
- **EDA_plots/**: Visualization outputs from EDA
- **outputs/**: General model outputs and results
- **results/**: Experiment results and performance metrics
- **saved_models/**: Trained model artifacts and metadata

## 5. Dataset Description

### Dataset Source
The system uses an exoplanet habitability dataset containing astronomical measurements and planetary characteristics relevant to habitability assessment.

### Features Used for Habitability Prediction
Based on the model schema and feature mapping:

- **P_RADIUS**: Planet radius in Earth radii
- **P_MASS**: Planet mass in Earth masses  
- **P_PERIOD**: Orbital period in days
- **S_TEMPERATURE**: Stellar temperature in Kelvin
- **S_LUMINOSITY**: Stellar luminosity in solar luminosities
- **P_DENSITY**: Planet density in g/cm³
- **P_SEMI_MAJOR_AXIS**: Semi-major axis in astronomical units (AU)

### Target Variable
- **P_HABITABLE**: Habitability classification with three classes:
  - 0: Non-Habitable
  - 1: Habitable  
  - 2: Likely Habitable

### Data Preprocessing Steps
1. **Missing Value Handling**: Columns with >40% missing values are dropped; numerical features filled with median values; categorical features filled with mode values
2. **Feature Selection**: Only numerical features are retained for model training
3. **Data Type Conversion**: Ensures consistent data types (int64, float64) for numerical features
4. **Dataset Splitting**: 80-20 train-test split with stratification to preserve class distribution

## 6. Machine Learning Pipeline

### Data Ingestion
- Raw dataset loaded from CSV format
- Initial data quality assessment and missing value analysis
- Feature type identification and selection

### Feature Engineering
- **Scaling**: StandardScaler applied to normalize feature distributions
- **Feature Mapping**: User-friendly feature names mapped to model-expected feature names
- **Feature Ordering**: Consistent feature order maintained via JSON configuration

### Model Selection
Three classification algorithms implemented:

1. **Class-Weighted SVM**: 
   - RBF kernel with balanced class weights
   - Probability estimation enabled for confidence scores
   - One-vs-Rest decision function for multiclass classification

2. **XGBoost Classifier**:
   - Multi-class softmax objective
   - Log-loss evaluation metric
   - Gradient boosting for ensemble learning

3. **Balanced Random Forest**:
   - 300 estimators with balanced sampling
   - Handles class imbalance inherently

### Training Process
1. **Stratified K-Fold Cross-Validation**: 5-fold validation preserving class distribution
2. **SMOTE Application**: Synthetic minority oversampling applied to training data only
3. **Pipeline Integration**: Preprocessing and model training combined in scikit-learn pipelines
4. **Hyperparameter Configuration**: Models configured with class balancing techniques

### Evaluation Metrics
- **F1-Score (Macro)**: Balanced performance across all classes
- **Precision (Macro)**: Average precision across classes
- **Recall (Macro)**: Average recall across classes  
- **ROC-AUC (One-vs-Rest)**: Multi-class area under curve
- **Matthews Correlation Coefficient**: Balanced metric for imbalanced datasets
- **Confusion Matrices**: Class-wise prediction accuracy visualization

### Model Persistence
- **Joblib Serialization**: Complete pipeline serialized including preprocessing steps
- **Feature Order Preservation**: JSON configuration maintains feature input order
- **Model Artifacts**: Cached loading for efficient API serving

## 7. API / Application Workflow

### Input Format
API accepts JSON requests with the following structure:
```json
{
  "features": {
    "planet_radius": 1.2,
    "planet_mass": 1.1,
    "orbital_period": 365.25,
    "stellar_temperature": 5778,
    "stellar_luminosity": 1.0,
    "planet_density": 5.5,
    "semi_major_axis": 1.0
  }
}
```

### Prediction Flow
1. **Request Validation**: Pydantic schemas validate input data types and constraints
2. **Feature Mapping**: User-friendly names mapped to model feature names
3. **Feature Array Construction**: Features arranged in model-expected order with default values for missing features
4. **Model Loading**: Cached pipeline loaded via dependency injection
5. **Preprocessing**: Same transformations applied as during training
6. **Prediction Generation**: Model generates class prediction and probability scores
7. **Response Formatting**: Results formatted with class labels and confidence scores

### Output Format
API returns structured JSON response:
```json
{
  "habitability_class": "Habitable",
  "confidence": 0.847,
  "model": "Class-weighted SVM (3-class)"
}
```

## 8. Environment Setup

### Python Version
- Python 3.8 or higher required
- Tested with Python 3.12

### Required Libraries
Core dependencies from requirements.txt:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0  
- pydantic==2.5.0
- numpy>=1.26,<2.0
- scikit-learn==1.3.2
- joblib==1.3.2
- imbalanced-learn==0.11.0
- xgboost==2.0.3

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)  
source venv/bin/activate

# Install dependencies
pip install -r Backend/requirements.txt
```

### How to Run Locally

**Backend Server:**
```bash
cd Backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Frontend Development:**
```bash
cd frontend
npm install
npm start
```

**ML Pipeline Execution:**
Execute numbered scripts in sequence from 02 through 12 to reproduce the complete analysis pipeline.

## 9. Deployment Details

### Deployment Platform
Current configuration supports local development deployment. Production deployment platform not specified in current project state.

### Environment Variables
- **MODEL_PATH**: Override default model file location
- **FEATURE_ORDER_PATH**: Override default feature order configuration location

### Startup Commands
- **API Server**: `uvicorn app:app --host 0.0.0.0 --port 8000`
- **Development Server**: `uvicorn app:app --reload` for auto-reloading during development

## 10. Limitations

### Known Constraints
- **Feature Scope**: Limited to seven planetary and stellar features
- **Class Imbalance**: Original dataset may have uneven distribution across habitability classes
- **Model Complexity**: Current models may not capture complex non-linear relationships between all astronomical factors

### Dataset Limitations
- **Feature Coverage**: May not include all relevant factors for habitability assessment
- **Data Quality**: Dependent on accuracy of astronomical observations and measurements
- **Sample Size**: Limited by available confirmed exoplanet data

### Model Limitations
- **Generalization**: Performance may vary on exoplanets with characteristics outside training data range
- **Uncertainty Quantification**: Confidence scores based on probability estimates may not reflect true prediction uncertainty
- **Feature Dependencies**: Assumes independence between features which may not hold in astronomical contexts

## 11. Future Enhancements

### Possible Improvements
- **Feature Expansion**: Incorporate additional astronomical features such as atmospheric composition, magnetic field strength, and orbital eccentricity
- **Advanced Models**: Implement deep learning models for capturing complex feature interactions
- **Ensemble Methods**: Combine multiple model predictions for improved accuracy and robustness

### Scalability Ideas
- **Batch Processing**: Support for bulk prediction requests
- **Model Versioning**: Implement model version management for continuous improvement
- **Real-time Updates**: Integration with astronomical databases for automatic model retraining

### Advanced ML Enhancements
- **Uncertainty Quantification**: Implement Bayesian approaches for better confidence estimation
- **Active Learning**: Incorporate feedback mechanisms for continuous model improvement
- **Multi-task Learning**: Extend to predict multiple habitability-related properties simultaneously
- **Explainable AI**: Add model interpretability features to understand prediction reasoning

## 12. Conclusion

The Exoplanet Habitability Prediction System successfully demonstrates the application of machine learning techniques to astronomical classification problems. The system addresses class imbalance through SMOTE, implements multiple classification algorithms for comparison, and provides a production-ready API interface for real-world usage.

The modular architecture allows for easy extension and improvement, while the comprehensive evaluation framework ensures reliable performance assessment. The system serves as a foundation for more advanced habitability prediction models and can be adapted for other astronomical classification tasks.

The technical implementation follows best practices for machine learning deployment, including proper data validation, model serialization, and API design, making it suitable for integration into larger astronomical research workflows.