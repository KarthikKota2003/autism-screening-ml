import pandas as pd
import numpy as np
import joblib
import argparse
import os
import json

def load_artifacts(model_path):
    """Load saved model artifacts"""
    print(f"Loading model artifacts from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    artifacts = joblib.load(model_path)
    return artifacts

def preprocess_input(df, artifacts):
    """Preprocess new input data using saved artifacts"""
    # Extract artifacts
    imputer = artifacts.get('imputer')
    ohe = artifacts.get('ohe')
    te = artifacts.get('te')
    scaler = artifacts['scaler']
    numerical_cols = artifacts.get('numerical_cols', [])
    ohe_cols = artifacts.get('ohe_cols', [])
    te_cols = artifacts.get('te_cols', [])
    
    # 1. Handle numerical columns
    if numerical_cols:
        # Ensure all numerical columns exist
        for col in numerical_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        if imputer:
            X_num = pd.DataFrame(
                imputer.transform(df[numerical_cols]),
                columns=numerical_cols,
                index=df.index
            )
        else:
            X_num = df[numerical_cols].fillna(0) # Fallback
    else:
        X_num = pd.DataFrame(index=df.index)
        
    # 2. Handle One-Hot Encoding
    if ohe and ohe_cols:
        # Ensure columns exist
        for col in ohe_cols:
            if col not in df.columns:
                df[col] = 'unknown'
                
        X_ohe = pd.DataFrame(
            ohe.transform(df[ohe_cols]),
            columns=ohe.get_feature_names_out(ohe_cols),
            index=df.index
        )
    else:
        X_ohe = pd.DataFrame(index=df.index)
        
    # 3. Handle Target Encoding
    if te and te_cols:
        for col in te_cols:
            if col not in df.columns:
                df[col] = 'unknown'
                
        X_te = te.transform(df[te_cols])
    else:
        X_te = pd.DataFrame(index=df.index)
        
    # Combine features
    X_processed = pd.concat([
        X_num.reset_index(drop=True),
        X_ohe.reset_index(drop=True),
        X_te.reset_index(drop=True)
    ], axis=1)
    
    # Scale features
    X_scaled = scaler.transform(X_processed)
    
    return X_scaled

def predict(data_path, model_path, output_path=None):
    """Main prediction function"""
    # Load artifacts
    artifacts = load_artifacts(model_path)
    model = artifacts['model']
    
    # Load data
    print(f"Loading data from: {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
        
    print(f"Input shape: {df.shape}")
    
    # Preprocess
    print("Preprocessing data...")
    try:
        X_scaled = preprocess_input(df, artifacts)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Ensure input data has same columns as training data.")
        return
        
    # Predict
    print("Running inference...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Create results DataFrame
    results = df.copy()
    results['Predicted_Class'] = predictions
    results['ASD_Probability'] = probabilities
    results['Prediction'] = results['Predicted_Class'].map({0: 'NO', 1: 'YES'})
    
    # Output
    print("\nPredictions:")
    print(results[['Predicted_Class', 'Prediction', 'ASD_Probability']].head())
    
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autism Prediction Inference')
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV/JSON')
    parser.add_argument('--model', type=str, required=True, help='Path to .pkl model file')
    parser.add_argument('--output', type=str, help='Path to save predictions CSV')
    
    args = parser.parse_args()
    
    predict(args.data, args.model, args.output)
