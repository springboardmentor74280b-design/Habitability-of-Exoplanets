Predicting Habitability of Exoplanets Using Machine Learning

Abstract
With the rapid advancement in space exploration, thousands of exoplanets have been discovered outside our solar system. Identifying potentially habitable exoplanets from this large dataset is a challenging task. This project applies Machine Learning techniques to predict the habitability of exoplanets based on their physical and orbital characteristics.

The project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and result interpretation.  
This work is carried out as part of the Infosys Internship / Project-Based Learning Program.

Problem Statement
Manual analysis of exoplanet data to determine habitability is complex, time-consuming, and inefficient due to the large volume of data and multiple influencing parameters.  
An automated system is required to accurately predict whether an exoplanet is potentially habitable using machine learning models.

Project Objectives
- To analyze real-world astronomical datasets
- To clean and preprocess datasets with missing and noisy values
- To apply supervised machine learning algorithms for classification
- To compare model performances and select the best model
- To evaluate results using standard performance metrics
- To document and present the project professionally for academic and industrial evaluation

Dataset Description
- Dataset Source: NASA Exoplanet Archive / Kaggle
- Initial Records: 4500+ exoplanets
- Processed Records: Reduced after cleaning and feature selection
- Key Features Used:
  - Planet Radius
  - Planet Mass
  - Orbital Period
  - Equilibrium Temperature
  - Stellar Radius
  - Stellar Luminosity

The dataset contains missing values and inconsistencies, making it suitable for real-world data preprocessing and analysis.

Technologies & Tools Used
Programming Language
- Python

Libraries
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

Tools & Platforms
- Jupyter Notebook
- GitHub

 Machine Learning Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

 Model Selection Criteria
- Accuracy
- Precision
- Recall
- F1-Score

Among the implemented models, the **Random Forest Classifier** provided the best overall performance and was selected as the final model.

System Workflow
1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection and Scaling
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Result Visualization
9. Model Saving

Results and Evaluation
- The Random Forest model achieved the highest accuracy compared to other models
- Performance evaluated using:
  - Accuracy Score
  - Confusion Matrix
  - Precision, Recall, and F1-Score
- Visual representations include:
  - Confusion Matrix
  - Feature Importance Graph

All result files and plots are available in the results directory.
