import shap 
import pandas as pd
import numpy as np

# Global Explainer Cache
EXPLAINER = None

def setup_explainability(pipeline, X_train_sample):
    """
    Initializes the SHAP explainer using the trained model.
    Must be called once when the app starts.
    """
    global EXPLAINER
    try:
        print("🧠 Initializing AI Explainability Engine (SHAP)...")
        
        # 1. Extract the actual AI model (XGBoost) from the pipeline
        # The pipeline is [Scaler, Model]. We need the Model (step -1).
        model = pipeline.steps[-1][1]
        
        # 2. Create the TreeExplainer
        # We don't pass the scaler here because TreeExplainer looks at the raw trees.
        # But we MUST pass scaled data when we ask for explanations later.
        EXPLAINER = shap.TreeExplainer(model)
        print("✅ Explainability Engine Ready.")
        
    except Exception as e:
        print(f"❌ Error setting up SHAP: {e}")

def explain_prediction(pipeline, input_df):
    """
    Explains WHY the model made a specific decision for a single planet.
    """
    global EXPLAINER
    
    if EXPLAINER is None:
        return []

    try:
        # 1. Feature Names (Must match training)
        feature_names = ['P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
                         'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 
                         'S_RADIUS', 'S_MASS', 'S_METALLICITY']
        
        # 2. Get Raw Data
        X_current = input_df[feature_names].fillna(0)
        
        # 3. Apply Transformations (Scaler) but Skip SMOTE
        for name, step in pipeline.steps[:-1]:
            if hasattr(step, 'transform'):
                X_current = step.transform(X_current)
            
        # 4. Calculate SHAP Values
        shap_values = EXPLAINER.shap_values(X_current)
        
        # 5. Extract the specific values for the "Habitable" class
        # Handle Binary vs Multiclass Output
        if isinstance(shap_values, list):
            # If list, it has one array per class. We want Class 1 (Habitable).
            # [1] selects class 1. [0] selects the first (and only) row.
            vals = shap_values[1][0] 
        else:
            # If single array, just take the first row.
            vals = shap_values[0]

        # 6. Safe Flattening (The Fix for "length-1 arrays" error)
        # Ensure vals and raw_values are simple 1D lists of numbers
        vals = np.ravel(vals) 
        raw_values = np.ravel(input_df[feature_names].fillna(0).iloc[0].values)
        
        # 7. Match values to features
        contributions = zip(feature_names, vals, raw_values)
        
        reasons = []
        for feature, impact, raw_val in contributions:
            # Safe extraction using .item() if it's a numpy type
            safe_impact = impact.item() if hasattr(impact, 'item') else float(impact)
            safe_val = raw_val.item() if hasattr(raw_val, 'item') else float(raw_val)

            reasons.append({
                "feature": feature,
                "impact": safe_impact,
                "value": safe_val
            })
            
        reasons.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return reasons[:5]

    except Exception as e:
        print(f"⚠️ SHAP Error: {e}")
        return []