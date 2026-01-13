"""
Script to train the Random Forest model for planet habitability prediction
Run this script to train/retrain the model before starting the Flask server
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model():
    """Train Random Forest model on Kepler data"""
    
    # Check if CSV file exists (try multiple possible locations)
    csv_path = '../Kepler_Threshold_Crossing_Events_Table.csv'
    if not os.path.exists(csv_path):
        csv_path = 'Kepler_Threshold_Crossing_Events_Table.csv'
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        print("Please ensure the CSV file is in the parent directory or backend directory")
        return False
    
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv(csv_path, comment='#', low_memory=False)
        print(f"Loaded {len(df)} rows")
        
        # Select relevant features for habitability prediction
        feature_columns = [
            'koi_period',           # Orbital period
            'koi_impact',           # Impact parameter
            'koi_duration',         # Transit duration
            'koi_depth',            # Transit depth
            'koi_prad',             # Planetary radius (Earth radii)
            'koi_teq',              # Equilibrium temperature
            'koi_insol',            # Insolation flux
            'koi_slogg',            # Stellar surface gravity
            'koi_srad',             # Stellar radius
            'koi_steff'             # Stellar effective temperature
        ]
        
        # Filter out rows with missing values in key columns
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Using features: {available_features}")
        
        df_clean = df[available_features + ['koi_disposition']].dropna()
        print(f"After cleaning: {len(df_clean)} rows")
        
        if len(df_clean) == 0:
            print("No valid data after cleaning")
            return False
        
        X = df_clean[available_features]
        
        # Create target variable: habitability score (0-100)
        print("Calculating habitability scores...")
        y = np.zeros(len(df_clean))
        
        for idx, row in df_clean.iterrows():
            score = 0
            
            # Temperature score (optimal: 200-320K)
            if 'koi_teq' in available_features and pd.notna(row['koi_teq']):
                temp = row['koi_teq']
                if 200 <= temp <= 320:
                    temp_score = 1.0 - abs(temp - 288) / 88
                    score += max(0, temp_score) * 40
            
            # Radius score (optimal: 0.5-2.5 Earth radii)
            if 'koi_prad' in available_features and pd.notna(row['koi_prad']):
                radius = row['koi_prad']
                if 0.5 <= radius <= 2.5:
                    rad_score = 1.0 - abs(radius - 1.0) / 1.5
                    score += max(0, rad_score) * 30
            
            # Insolation score (optimal: 0.25-2.0)
            if 'koi_insol' in available_features and pd.notna(row['koi_insol']):
                insol = row['koi_insol']
                if 0.25 <= insol <= 2.0:
                    insol_score = 1.0 - abs(insol - 1.0) / 1.0
                    score += max(0, insol_score) * 20
            
            # Period score (optimal: 50-500 days)
            if 'koi_period' in available_features and pd.notna(row['koi_period']):
                period = row['koi_period']
                if 50 <= period <= 500:
                    period_score = 1.0 - abs(period - 365) / 315
                    score += max(0, period_score) * 10
            
            y[idx] = min(100, max(0, score))
        
        print(f"Score range: {y.min():.2f} - {y.max():.2f}")
        print(f"Mean score: {y.mean():.2f}")
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest model
        print("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"\nModel Performance:")
        print(f"Training R²: {train_score:.4f}")
        print(f"Testing R²: {test_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop Features by Importance:")
        print(feature_importance.head(10))
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        model_path = 'models/random_forest_model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
        
        print("\nModel training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = train_model()
    sys.exit(0 if success else 1)
