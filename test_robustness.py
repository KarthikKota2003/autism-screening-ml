import pandas as pd
import joblib
import os
import numpy as np

def test_robustness():
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
    
    # Helper function to predict
    def predict_row(row_data, description):
        print(f"\n--- {description} ---")
        df = pd.DataFrame(row_data)
        
        # 1. Handle Categorical Missing Values (Fill with placeholder)
        for col in ohe_cols:
            if col in df.columns:
                df[col] = df[col].fillna('missing')
        
        # 2. Impute Numerical
        if imputer:
            df_num = pd.DataFrame(
                imputer.transform(df[numerical_cols]),
                columns=numerical_cols
            )
        else:
            df_num = df[numerical_cols]
            
        # 3. OHE
        if ohe:
            df_ohe = pd.DataFrame(
                ohe.transform(df[ohe_cols]),
                columns=ohe.get_feature_names_out(ohe_cols)
            )
        else:
            df_ohe = pd.DataFrame()
            
        # 4. TE
        if te:
            # TargetEncoder handles NaNs automatically (usually)
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
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        
        print(f"Prediction: {pred} (0=NO, 1=YES)")
        print(f"Probability: {prob}")
        return pred, prob

    # Base Data (Class NO)
    base_data = {
        'A1_Score': [1], 'A2_Score': [1], 'A3_Score': [0], 'A4_Score': [0], 'A5_Score': [1],
        'A6_Score': [1], 'A7_Score': [0], 'A8_Score': [1], 'A9_Score': [0], 'A10_Score': [0],
        'age': [6.0],
        'gender': ['m'], 'ethnicity': ['others'], 'jaundice': ['no'], 'family_asd': ['no'],
        'contry_of_res': ['Jordan'], 'used_app_before': ['no'], 'relation': ['parent']
    }

    # Scenario 1: All Missing
    all_missing = {k: [np.nan] for k in base_data.keys()}
    predict_row(all_missing, "SCENARIO 1: ALL PARAMETERS MISSING")

    # Scenario 2: Missing Scores (A1-A10) - "Mandatory" candidates
    missing_scores = base_data.copy()
    for i in range(1, 11):
        missing_scores[f'A1{i}_Score' if i==10 else f'A{i}_Score'] = [np.nan]
    predict_row(missing_scores, "SCENARIO 2: MISSING ALL SCORES (A1-A10)")

    # Scenario 3: Missing Demographics - "Non-Mandatory" candidates
    missing_demographics = base_data.copy()
    for col in ['age', 'gender', 'ethnicity', 'jaundice', 'family_asd', 'contry_of_res', 'used_app_before', 'relation']:
        missing_demographics[col] = [np.nan]
    predict_row(missing_demographics, "SCENARIO 3: MISSING ALL DEMOGRAPHICS")

if __name__ == "__main__":
    test_robustness()
