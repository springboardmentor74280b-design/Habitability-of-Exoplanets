import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Load data
# -----------------------------
from preprocessing.data_loader import load_data
from preprocessing.cleaning import drop_useless_columns, impute_missing
from preprocessing.outlier_iqr import iqr_outlier_capping
# from preprocessing.encoding import frequency_encode (Removed)
from preprocessing.scaling import standardize
from preprocessing.pca import apply_pca
from preprocessing.smote import apply_smote

# Models
from models.logistic_regression import train_logistic
from models.svm_linear import train_linear_svm
from models.svm_rbf import train_rbf_svm
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost

# Evaluation
from evaluation.metrics import evaluate
from evaluation.shap_analysis import shap_analysis
from visualization.tsne_plot import plot_tsne


# -----------------------------
# CONFIG
# -----------------------------
TARGET_COL = "P_HABITABLE"

DROP_COLS = [
    "P_NAME", "S_NAME", "S_NAME_HD", "S_NAME_HIP",
    "S_RA_STR", "S_DEC_STR", "S_RA_TXT", "S_DEC_TXT",
    "P_UPDATE",
    # Leakage Columns (Directly correlated with Target)
    "P_ESI",            # Earth Similarity Index (Answer Key)
    "P_TYPE",           # E.g. "Terran" (Gives it away)
    "P_TYPE_TEMP",      # E.g. "Warm" (Gives it away)
    "P_HABZONE_OPT",    # Habitable Zone Optimistic
    "P_HABZONE_CON",    # Habitable Zone Conservative
]

USE_PCA = True        # PCA pipeline (SVM RBF)
USE_SMOTE = True      # SMOTE only on training
RUN_TSNE = True      # Visualization only
RUN_SHAP = False      # Only for RF / XGBoost


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():

    # 1. Load
    # FIX: Use relative path from ml_pipeline pointing to Dataset
    df = load_data("../Dataset/hwc.csv")

    # 2. Drop useless columns
    df = drop_useless_columns(df, DROP_COLS)

    # 3. Separate X / y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 4. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 5. Identify column types
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # 6. Impute missing values (FIT ON TRAIN)
    X_train = impute_missing(X_train, num_cols, cat_cols)
    X_test = impute_missing(X_test, num_cols, cat_cols)

    # 7. IQR outlier handling (numerical only)
    X_train = iqr_outlier_capping(X_train, num_cols)
    X_test = iqr_outlier_capping(X_test, num_cols)

    # 8. Hybrid Encoding (OHE for Low Card, Freq for High Card)
    from preprocessing.encoding import hybrid_encode
    # Returns updated DFs and the list of columns that were freq encoded
    # UPDATED: Unpack new return values
    X_train, X_test, freq_encoded_cols, freq_maps, ohe_encoder = hybrid_encode(X_train, X_test, cat_cols, threshold=10)

    # 9. Standardization
    # We scale: Original Numerical Cols + Frequency Encoded Cols
    # We do NOT scale: One-Hot Encoded Cols
    cols_to_scale = num_cols + freq_encoded_cols
    
    # Ensure cols exist (in case some were dropped or modified, though they shouldn't be)
    cols_to_scale = [c for c in cols_to_scale if c in X_train.columns]
    
    X_train_scaled, X_test_scaled, scaler_obj = standardize(X_train, X_test, cols_to_scale)

    # 10. SMOTE (TRAIN ONLY)
    # Apply SMOTE *before* PCA.
    # Note: imblearn usually returns DataFrames if input is DataFrame.
    if USE_SMOTE:
        X_train_scaled, y_train = apply_smote(X_train_scaled, y_train)
        
    # Update feature names for importance analysis (Post-Encoding/OHE)
    if hasattr(X_train_scaled, "columns"):
        feature_names = X_train_scaled.columns.tolist()
    else:
        # If it returned an array, we try to use the ones from before SMOTE
        # (Assuming SMOTE didn't add columns, which it doesn't)
        feature_names = X_train.columns.tolist()

    # 11. PCA (optional – for SVM RBF)
    if USE_PCA:
        X_train_final, X_test_final, pca = apply_pca(X_train_scaled, X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------

    # print("\n=== Logistic Regression ===")
    # lr = train_logistic(X_train_scaled, y_train)
    # evaluate(lr, X_test_scaled, y_test)

    # print("\n=== Linear SVM ===")
    # svm_linear = train_linear_svm(X_train_scaled, y_train)
    # evaluate(svm_linear, X_test_scaled, y_test)

    # print("\n=== SVM (RBF + PCA) ===")
    # svm_rbf = train_rbf_svm(X_train_final, y_train)
    # evaluate(svm_rbf, X_test_final, y_test)

    # print("\n=== Random Forest ===")
    # rf = train_random_forest(X_train_scaled, y_train)
    # evaluate(rf, X_test_scaled, y_test)

    print("\n=== XGBoost (Best Model) ===")
    xgb = train_xgboost(X_train_scaled, y_train)
    evaluate(xgb, X_test_scaled, y_test)

    # -----------------------------
    # SAVING ARTIFACTS
    # -----------------------------
    import joblib
    import os
    import json

    save_dir = "../models_saved"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving artifacts to {save_dir}...")
    
    # 1. Model
    joblib.dump(xgb, os.path.join(save_dir, "xgboost_model.pkl"))
    
    # 2. Scaler
    joblib.dump(scaler_obj, os.path.join(save_dir, "scaler.pkl"))
    
    # 3. Encoding Artifacts
    joblib.dump(freq_maps, os.path.join(save_dir, "freq_maps.pkl"))
    joblib.dump(ohe_encoder, os.path.join(save_dir, "ohe_encoder.pkl"))
    
    # 4. Feature Names (Crucial for verifying input)
    # Note: We save the columns expected AFTER encoding but BEFORE scaling (same for this pipeline)
    # Actually, we want the columns expected by the model. 
    # Use feature_names list.
    with open(os.path.join(save_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
        
    # 5. Cols to Scale (List)
    with open(os.path.join(save_dir, "cols_to_scale.json"), "w") as f:
        json.dump(cols_to_scale, f)

    # 6. Frequency Encoded Feature Names (List)
    with open(os.path.join(save_dir, "freq_encoded_cols.json"), "w") as f:
        json.dump(freq_encoded_cols, f)

    print("Artifacts saved successfully.")

    # -----------------------------
    # PLOTTING
    # -----------------------------
    from visualization.confusion_matrix import plot_model_results
    
    # Just Plot XGBoost for now to save time/confusion
    results_xgb = {"XGBoost": (y_test, xgb.predict(X_test_scaled))}
    # Fix path for outputs too
    output_dir = "../Outputs"
    os.makedirs(output_dir, exist_ok=True)
    plot_model_results(results_xgb, os.path.join(output_dir, "xgboost_confusion_matrix.png"))

    # -----------------------------
    # OPTIONAL ANALYSIS
    # -----------------------------
    if RUN_SHAP:
        shap_analysis(xgb, X_train_scaled)

    if RUN_TSNE:
        plot_tsne(X_train_scaled, y_train, save_path=os.path.join(output_dir, "tsne_plot.png"))


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()