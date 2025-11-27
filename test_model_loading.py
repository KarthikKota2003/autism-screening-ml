import pandas as pd
import joblib
import os
import numpy as np

def test_model():
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
    print(f"Scaler loaded: {artifacts['scaler_name']}")
    
    # Create a sample input (based on row 2 of dataset, which is Class/ASD='NO')
    # 1,1,1,0,0,1,1,0,1,0,0,6.0,m,others,no,no,Jordan,no,5,parent,NO
    sample_data = {
        'A1_Score': [1], 'A2_Score': [1], 'A3_Score': [0], 'A4_Score': [0], 'A5_Score': [1],
        'A6_Score': [1], 'A7_Score': [0], 'A8_Score': [1], 'A9_Score': [0], 'A10_Score': [0],
        'age': [6.0],
        'gender': ['m'], 'ethnicity': ['others'], 'jaundice': ['no'], 'family_asd': ['no'],
        'contry_of_res': ['Jordan'], 'used_app_before': ['no'], 'relation': ['parent']
    }
    
    df = pd.DataFrame(sample_data)
    print("\nSample Input:")
    print(df)
    
    # 1. Impute numerical columns
    if imputer:
        df_num = pd.DataFrame(
            imputer.transform(df[numerical_cols]),
            columns=numerical_cols
        )
    else:
        df_num = df[numerical_cols]
        
    # 2. One-Hot Encoding
    if ohe:
        df_ohe = pd.DataFrame(
            ohe.transform(df[ohe_cols]),
            columns=ohe.get_feature_names_out(ohe_cols)
        )
    else:
        df_ohe = pd.DataFrame()
        
    # 3. Target Encoding
    if te:
        df_te = te.transform(df[te_cols])
    else:
        df_te = pd.DataFrame()
        
    # 4. Combine
    X_processed = pd.concat([
        df_num.reset_index(drop=True),
        df_ohe.reset_index(drop=True),
        df_te.reset_index(drop=True)
    ], axis=1)
    
    print(f"\nProcessed feature shape: {X_processed.shape}")
    
    # 5. Scale
    X_scaled = scaler.transform(X_processed)
    
    # 6. Predict
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    
    print(f"\nPrediction: {prediction[0]} (0=NO, 1=YES)")
    print(f"Probability: {probability[0]}")
    
    # Expected result for this sample is NO (0)
    print(f"\nExpected: 0 (NO)")
    if prediction[0] == 0:
        print("SUCCESS: Prediction matches expectation.")
    else:
        print("WARNING: Prediction does not match expectation (this might be due to model performance or sample choice).")

if __name__ == "__main__":
    test_model()
