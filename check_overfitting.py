import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessing.data_loader import load_data
from preprocessing.cleaning import drop_useless_columns, impute_missing
from preprocessing.outlier_iqr import iqr_outlier_capping
from preprocessing.encoding import frequency_encode
from preprocessing.scaling import standardize
from models.xgboost_model import train_xgboost
import xgboost as xgb

# CONFIG
TARGET_COL = "P_HABITABLE"
DROP_COLS = [
    "P_NAME", "S_NAME", "S_NAME_HD", "S_NAME_HIP",
    "S_RA_STR", "S_DEC_STR", "S_RA_TXT", "S_DEC_TXT",
    "P_UPDATE",
    # Leakage Columns
    "P_ESI", "P_TYPE", "P_TYPE_TEMP", "P_HABZONE_OPT", "P_HABZONE_CON"
]

def check_overfitting():
    print("--- Loading and Preprocessing Data ---")
    df = load_data("Dataset/hwc.csv")
    df = drop_useless_columns(df, DROP_COLS)
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Preprocessing (Replicating main.py pipeline)
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    X_train = impute_missing(X_train, num_cols, cat_cols)
    X_test = impute_missing(X_test, num_cols, cat_cols)
    
    X_train = iqr_outlier_capping(X_train, num_cols)
    X_test = iqr_outlier_capping(X_test, num_cols)
    
    X_train = frequency_encode(X_train, cat_cols)
    X_test = frequency_encode(X_test, cat_cols)
    
    X_train_scaled, X_test_scaled, scaler = standardize(X_train, X_test)
    
    # 1. Compare Train vs Test Accuracy
    print("\n--- 1. Train vs Test Performance ---")
    model = train_xgboost(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    
    if train_acc > 0.99 and test_acc < 0.90:
        print(">> WARNING: High variance detected (Likely Overfitting)")
    elif abs(train_acc - test_acc) < 0.02:
        print(">> GOOD: Train and Test accuracies are similar (Good Generalization)")
    else:
        print(">> INFO: Some gap between train and test.")

    # 2. Cross Validation (Robustness Check)
    print("\n--- 2. 5-Fold Cross Validation ---")
    # We need to preprocess the entire dataset for CV or use a pipeline. 
    # For simplicity, we'll use the already scaled training set which is large enough (80%) to proxy.
    # Ideally should pipeline inside CV, but this gives a good enough signal.
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Note: re-instantiating model to be safe
    cv_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    print(f"CV Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy:   {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    if scores.min() < 0.90:
        print(">> WARNING: Some folds have significantly lower accuracy. Model might be unstable.")
    else:
        print(">> GOOD: High accuracy across all folds.")

if __name__ == "__main__":
    check_overfitting()
