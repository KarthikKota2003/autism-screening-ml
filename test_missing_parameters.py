"""
Test script to analyze the impact of missing parameters on the ML pipeline.
This script systematically removes parameters and measures prediction quality.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """Load the test set from the adolescent dataset"""
    # Load full dataset
    df = pd.read_csv('Autism_Adolescent_Data_Preprocessed.csv')
    
    # Drop leakage columns
    drop_cols = ['Class/ASD', 'result'] + [f'A{i}_Score' for i in range(1, 11)]
    
    # Separate features and target
    X = df.drop(drop_cols, axis=1)
    y = (df['Class/ASD'] == 'YES').astype(int)
    
    return X, y

def preprocess_and_predict(X, artifacts):
    """Preprocess data and make predictions using saved artifacts"""
    # Extract artifacts
    imputer = artifacts.get('imputer')
    ohe = artifacts.get('ohe')
    te = artifacts.get('te')
    scaler = artifacts['scaler']
    model = artifacts['model']
    numerical_cols = artifacts.get('numerical_cols', [])
    ohe_cols = artifacts.get('ohe_cols', [])
    te_cols = artifacts.get('te_cols', [])
    
    # Handle numerical columns
    if numerical_cols:
        for col in numerical_cols:
            if col not in X.columns:
                X[col] = np.nan
        
        if imputer:
            X_num = pd.DataFrame(
                imputer.transform(X[numerical_cols]),
                columns=numerical_cols,
                index=X.index
            )
        else:
            X_num = X[numerical_cols].fillna(0)
    else:
        X_num = pd.DataFrame(index=X.index)
    
    # Handle One-Hot Encoding
    if ohe and ohe_cols:
        for col in ohe_cols:
            if col not in X.columns:
                X[col] = 'unknown'
        
        X_ohe = pd.DataFrame(
            ohe.transform(X[ohe_cols]),
            columns=ohe.get_feature_names_out(ohe_cols),
            index=X.index
        )
    else:
        X_ohe = pd.DataFrame(index=X.index)
    
    # Handle Target Encoding
    if te and te_cols:
        for col in te_cols:
            if col not in X.columns:
                X[col] = 'unknown'
        
        X_te = te.transform(X[te_cols])
    else:
        X_te = pd.DataFrame(index=X.index)
    
    # Combine and scale
    X_processed = pd.concat([
        X_num.reset_index(drop=True),
        X_ohe.reset_index(drop=True),
        X_te.reset_index(drop=True)
    ], axis=1)
    
    X_scaled = scaler.transform(X_processed)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities

def test_missing_parameter(X, y, artifacts, param_to_remove):
    """Test prediction quality when a specific parameter is missing"""
    X_test = X.copy()
    
    # Remove the parameter
    if param_to_remove in X_test.columns:
        X_test = X_test.drop(columns=[param_to_remove])
    
    try:
        predictions, probabilities = preprocess_and_predict(X_test, artifacts)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        roc_auc = roc_auc_score(y, probabilities)
        f1 = f1_score(y, predictions)
        
        return {
            'parameter': param_to_remove,
            'status': 'SUCCESS',
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'error': None
        }
    except Exception as e:
        return {
            'parameter': param_to_remove,
            'status': 'ERROR',
            'accuracy': None,
            'roc_auc': None,
            'f1_score': None,
            'error': str(e)
        }

def main():
    print("="*80)
    print("PARAMETER REQUIREMENT ANALYSIS")
    print("="*80)
    
    # Load model artifacts
    model_path = 'results/adolescent_no_leakage/adolescent_best_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"\nLoading model from: {model_path}")
    artifacts = joblib.load(model_path)
    
    # Load test data
    print("Loading test data...")
    X, y = load_test_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Baseline prediction (all parameters present)
    print("\n" + "="*80)
    print("BASELINE (All Parameters Present)")
    print("="*80)
    baseline_preds, baseline_probs = preprocess_and_predict(X.copy(), artifacts)
    baseline_accuracy = accuracy_score(y, baseline_preds)
    baseline_roc_auc = roc_auc_score(y, baseline_probs)
    baseline_f1 = f1_score(y, baseline_preds)
    
    print(f"Accuracy:  {baseline_accuracy:.4f}")
    print(f"ROC-AUC:   {baseline_roc_auc:.4f}")
    print(f"F1-Score:  {baseline_f1:.4f}")
    
    # Define parameter categories
    mandatory_params = ['age', 'gender', 'ethnicity', 'jaundice', 'family_asd']
    optional_params = ['contry_of_res', 'used_app_before', 'relation']
    
    # Test removing each mandatory parameter
    print("\n" + "="*80)
    print("TESTING MANDATORY PARAMETERS")
    print("="*80)
    
    mandatory_results = []
    for param in mandatory_params:
        print(f"\nTesting without: {param}")
        result = test_missing_parameter(X, y, artifacts, param)
        mandatory_results.append(result)
        
        if result['status'] == 'SUCCESS':
            acc_diff = result['accuracy'] - baseline_accuracy
            auc_diff = result['roc_auc'] - baseline_roc_auc
            print(f"  Status: {result['status']}")
            print(f"  Accuracy:  {result['accuracy']:.4f} (Delta {acc_diff:+.4f})")
            print(f"  ROC-AUC:   {result['roc_auc']:.4f} (Delta {auc_diff:+.4f})")
            print(f"  F1-Score:  {result['f1_score']:.4f}")
        else:
            print(f"  Status: {result['status']}")
            print(f"  Error: {result['error']}")
    
    # Test removing each optional parameter
    print("\n" + "="*80)
    print("TESTING OPTIONAL PARAMETERS")
    print("="*80)
    
    optional_results = []
    for param in optional_params:
        print(f"\nTesting without: {param}")
        result = test_missing_parameter(X, y, artifacts, param)
        optional_results.append(result)
        
        if result['status'] == 'SUCCESS':
            acc_diff = result['accuracy'] - baseline_accuracy
            auc_diff = result['roc_auc'] - baseline_roc_auc
            print(f"  Status: {result['status']}")
            print(f"  Accuracy:  {result['accuracy']:.4f} (Delta {acc_diff:+.4f})")
            print(f"  ROC-AUC:   {result['roc_auc']:.4f} (Delta {auc_diff:+.4f})")
            print(f"  F1-Score:  {result['f1_score']:.4f}")
        else:
            print(f"  Status: {result['status']}")
            print(f"  Error: {result['error']}")
    
    # Save results to CSV
    all_results = [
        {
            'parameter': 'BASELINE (All)',
            'category': 'Baseline',
            'status': 'SUCCESS',
            'accuracy': baseline_accuracy,
            'roc_auc': baseline_roc_auc,
            'f1_score': baseline_f1,
            'accuracy_delta': 0.0,
            'roc_auc_delta': 0.0,
            'error': None
        }
    ]
    
    for result in mandatory_results:
        all_results.append({
            'parameter': result['parameter'],
            'category': 'Mandatory',
            'status': result['status'],
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc'],
            'f1_score': result['f1_score'],
            'accuracy_delta': result['accuracy'] - baseline_accuracy if result['accuracy'] else None,
            'roc_auc_delta': result['roc_auc'] - baseline_roc_auc if result['roc_auc'] else None,
            'error': result['error']
        })
    
    for result in optional_results:
        all_results.append({
            'parameter': result['parameter'],
            'category': 'Optional',
            'status': result['status'],
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc'],
            'f1_score': result['f1_score'],
            'accuracy_delta': result['accuracy'] - baseline_accuracy if result['accuracy'] else None,
            'roc_auc_delta': result['roc_auc'] - baseline_roc_auc if result['roc_auc'] else None,
            'error': result['error']
        })
    
    results_df = pd.DataFrame(all_results)
    output_path = 'results/parameter_analysis.csv'
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print("\nKey Findings:")
    print(f"- Total parameters tested: {len(mandatory_params) + len(optional_params)}")
    print(f"- Mandatory parameters: {len(mandatory_params)}")
    print(f"- Optional parameters: {len(optional_params)}")

if __name__ == "__main__":
    main()
