import pandas as pd
import joblib
import os
import numpy as np

def predict_with_partial_input():
    # Path to saved model
    model_path = os.path.join('results', 'child_production', 'child_best_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    artifacts = joblib.load(model_path)
    
    model = artifacts['model']
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']
    ohe = artifacts['ohe']
    te = artifacts['te']
    numerical_cols = artifacts['numerical_cols']
    ohe_cols = artifacts['ohe_cols']
    te_cols = artifacts['te_cols']
    
    print(f"Model loaded: {artifacts['model_name']}")
    
    # Sample input with MISSING VALUES (NaNs)
    sample_data = {
        'A1_Score': [1], 
        'A2_Score': [np.nan], # Missing numerical (will be imputed)
        'A3_Score': [0], 
        'A4_Score': [0], 
        'A5_Score': [1],
        'A6_Score': [1], 
        'A7_Score': [0], 
        'A8_Score': [1], 
        'A9_Score': [0], 
        'A10_Score': [0],
        'age': [np.nan],      # Missing numerical (will be imputed)
        'gender': ['m'], 
        'ethnicity': [np.nan], # Missing categorical (needs handling)
        'jaundice': ['no'], 
        'family_asd': ['no'],
        'contry_of_res': ['Jordan'], 
        'used_app_before': ['no'], 
        'relation': ['parent']
    }
    
    df = pd.DataFrame(sample_data)
    print("\nInput Data (with NaNs):")
    print(df[['A2_Score', 'age', 'ethnicity']])
    
    # --- PREPROCESSING FOR INFERENCE ---
    
    # 1. Handle Categorical Missing Values
    # The pipeline's OHE expects strings, so we must fill NaNs with a placeholder.
    # The OHE is configured with handle_unknown='ignore', so it will gracefully 
    # ignore this 'missing' label (treating it as all-zeros).
    for col in ohe_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing')
            
    # 2. Impute Numerical Columns
    # The pipeline has a built-in imputer for this.
    if imputer:
        df_num = pd.DataFrame(
            imputer.transform(df[numerical_cols]),
            columns=numerical_cols
        )
    else:
        df_num = df[numerical_cols]
        
    # 3. One-Hot Encoding
    if ohe:
        df_ohe = pd.DataFrame(
            ohe.transform(df[ohe_cols]),
            columns=ohe.get_feature_names_out(ohe_cols)
        )
    else:
        df_ohe = pd.DataFrame()
        
    # 4. Target Encoding
    if te:
        df_te = te.transform(df[te_cols])
    else:
        df_te = pd.DataFrame()
        
    # 5. Combine
    X_processed = pd.concat([
        df_num.reset_index(drop=True),
        df_ohe.reset_index(drop=True),
        df_te.reset_index(drop=True)
    ], axis=1)
    
    # 6. Scale
    X_scaled = scaler.transform(X_processed)
    
    # 7. Predict
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    
    print(f"\nPrediction: {prediction[0]} (0=NO, 1=YES)")
    print(f"Probability: {probability[0]}")
    print("\nSUCCESS: Prediction generated despite missing inputs.")

if __name__ == "__main__":
    predict_with_partial_input()
