"""
Universal ML Pipeline for Autism Screening Datasets
====================================================

This script implements a comprehensive, leakage-safe ML pipeline that works
for all autism datasets (Adolescent, Adult, Child, Toddler).

Usage:
    python ml_pipeline.py --dataset toddler --output_dir results/toddler

Features:
    - Leakage-safe preprocessing (fit on train only)
    - 4 scaling methods: QT, PT, Normalizer, MAS
    - 8 ML algorithms: DT, KNN, LDA, GNB, LR, AB, RF, SVM
    - Hyperparameter tuning: Grid Search, Random Search, Bayesian Optimization
    - 8 evaluation metrics: Accuracy, ROC-AUC, F1, Precision, Recall, MCC, Kappa, Log Loss
    - 5-fold cross-validation
    - Comprehensive results logging
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import warnings
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import (
    OneHotEncoder, 
    QuantileTransformer, 
    PowerTransformer, 
    Normalizer, 
    MaxAbsScaler
)
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

# SMOTE
from imblearn.over_sampling import SMOTE

# ML Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

# Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss
)

# Scipy for Random Search
from scipy.stats import uniform, randint


class AutismMLPipeline:
    """Universal ML Pipeline for Autism Screening Datasets"""
    
    def __init__(self, dataset_name, random_state=42):
        """
        Initialize the pipeline
        
        Args:
            dataset_name: Name of dataset ('adolescent', 'adult', 'child', 'toddler')
            random_state: Random seed for reproducibility
        """
        self.dataset_name = dataset_name.lower()
        self.random_state = random_state
        self.results = []
        # Flags set via CLI
        self.evaluate_train = False
        self.apply_mitigation = False
        
        # Define scalers
        self.scalers = {
            'QuantileTransformer': QuantileTransformer(
                output_distribution='normal', 
                random_state=random_state
            ),
            'PowerTransformer': PowerTransformer(
                method='yeo-johnson', 
                standardize=True
            ),
            'Normalizer': Normalizer(norm='l2'),
            'MaxAbsScaler': MaxAbsScaler()
        }
        
        # Define models and their tuning methods
        self.models = self._get_models()
        self.param_grids = self._get_param_grids()
        
    def _get_models(self):
        """Define all ML models"""
        return {
            'DecisionTree': DecisionTreeClassifier(random_state=self.random_state),
            'KNN': KNeighborsClassifier(),
            'LDA': LinearDiscriminantAnalysis(),
            'GaussianNB': GaussianNB(),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'AdaBoost': AdaBoostClassifier(random_state=self.random_state),
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'SVM': SVC(probability=True, random_state=self.random_state)
        }
    
    def _get_param_grids(self):
        """Define hyperparameter grids for each model"""
        return {
            # Grid Search - Simple Models
            'DecisionTree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'LDA': {
                'solver': ['svd', 'lsqr', 'eigen']
            },
            'GaussianNB': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            # Random Search / Bayesian - Moderate Models
            'LogisticRegression': {
                'C': Real(0.01, 10, prior='log-uniform'),
                'penalty': Categorical(['l1', 'l2']),
                'solver': Categorical(['liblinear', 'saga'])
            },
            'AdaBoost': {
                'n_estimators': Integer(50, 300),
                'learning_rate': Real(0.01, 2.0, prior='log-uniform')
            },
            'RandomForest': {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(5, 30),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10)
            },
            # Bayesian - Complex Model
            'SVM': {
                'C': Real(0.1, 100, prior='log-uniform'),
                'gamma': Real(1e-4, 1e-1, prior='log-uniform'),
            }
        }
    
    def validate_input(self, df):
        """
        Validate input data for mandatory parameters
        
        Args:
            df: Input dataframe
            
        Raises:
            ValueError: If mandatory fields are missing or invalid
        """
        # Define mandatory fields (excluding A1-A10 and result which are dropped)
        mandatory_fields = {
            'gender': ['f', 'm'],
            'jaundice': ['yes', 'no'],
            'family_asd': ['yes', 'no'],
            'contry_of_res': None,  # Any country is valid
            'used_app_before': ['yes', 'no']
        }
        
        print(f"\nValidating mandatory input parameters:")
        errors = []
        warnings = []
        
        for field, valid_values in mandatory_fields.items():
            # Check if field exists
            if field not in df.columns:
                errors.append(f"Missing mandatory field: '{field}'")
                continue
            
            # Check for missing values (NaN or '?')
            missing_count = df[field].isnull().sum() + (df[field] == '?').sum()
            if missing_count > 0:
                warnings.append(f"'{field}' has {missing_count} missing values (will be imputed)")
            
            # Check for invalid values (if valid_values specified)
            if valid_values is not None:
                invalid_mask = ~df[field].isin(valid_values + ['?'])
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    invalid_vals = df[field][invalid_mask].unique()[:5]
                    warnings.append(f"'{field}' has {invalid_count} invalid values: {invalid_vals}")
        
        # Print validation results
        if errors:
            print(f"  [ERROR] Validation failed:")
            for error in errors:
                print(f"    - {error}")
            raise ValueError(f"Input validation failed: {errors}")
        
        if warnings:
            print(f"  [WARNING] Validation warnings:")
            for warning in warnings:
                print(f"    - {warning}")
        else:
            print(f"  [OK] All mandatory fields present and valid")
        
        return True
    
    def load_data(self, filepath):
        """Load preprocessed dataset"""
        print(f"\n{'='*80}")
        print(f"Loading {self.dataset_name} dataset from: {filepath}")
        print(f"{'='*80}")
        
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Validate input
        self.validate_input(df)
        
        return df
    
    def preprocess_data(self, df):
        """
        Leakage-safe preprocessing
        
        Returns:
            X_train, X_test, y_train, y_test (before scaling)
        """
        print(f"\n{'='*80}")
        print("PREPROCESSING - LEAKAGE-SAFE")
        print(f"{'='*80}")
        
        # Separate features and target
        target_col = 'Class/ASD'
        # Drop target AND 'result' (leakage)
        drop_cols = [target_col]
        if 'result' in df.columns:
            drop_cols.append('result')
            
        # Drop A1-A10 scores (leakage - they define the target)
        score_cols = [f'A{i}_Score' for i in range(1, 11)]
        drop_cols.extend([col for col in score_cols if col in df.columns])
        
        print(f"Dropping leakage columns: {drop_cols}")
            
        X = df.drop(drop_cols, axis=1)
        y = (df[target_col] == 'YES').astype(int)  # Label encode: YES=1, NO=0
        
        print(f"\nTarget distribution:")
        print(f"  Class 0 (NO): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
        print(f"  Class 1 (YES): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
        
        # Train/test split (BEFORE any encoding!)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain/Test split (80/20):")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        # Identify feature types
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove ID columns from numerical features
        id_cols = [col for col in numerical_cols if 'id' in col.lower() or 'case' in col.lower()]
        numerical_cols = [col for col in numerical_cols if col not in id_cols]
        
        print(f"\nFeature types:")
        print(f"  Categorical: {len(categorical_cols)} - {categorical_cols}")
        print(f"  Numerical: {len(numerical_cols)} - {numerical_cols}")
        print(f"  ID columns (excluded): {id_cols}")
        
        # FIX 1: Handle '?' values in categorical columns (replace with mode)
        print(f"\nHandling missing values in categorical columns:")
        for col in categorical_cols:
            # Count '?' values in training data
            question_count_train = (X_train[col] == '?').sum()
            question_count_test = (X_test[col] == '?').sum()
            
            if question_count_train > 0 or question_count_test > 0:
                # Calculate mode from non-missing training data
                valid_values = X_train[X_train[col] != '?'][col]
                if len(valid_values) > 0:
                    mode_value = valid_values.mode()[0]
                    print(f"  {col}: Replacing {question_count_train} train + {question_count_test} test '?' with mode '{mode_value}'")
                    X_train[col] = X_train[col].replace('?', mode_value)
                    X_test[col] = X_test[col].replace('?', mode_value)
                else:
                    # Fallback: replace with 'Unknown'
                    print(f"  {col}: Replacing {question_count_train} train + {question_count_test} test '?' with 'Unknown'")
                    X_train[col] = X_train[col].replace('?', 'Unknown')
                    X_test[col] = X_test[col].replace('?', 'Unknown')
        
        # FIX 2: Handle NaN values in categorical columns (safety measure)
        for col in categorical_cols:
            nan_count_train = X_train[col].isnull().sum()
            nan_count_test = X_test[col].isnull().sum()
            
            if nan_count_train > 0 or nan_count_test > 0:
                # Use mode imputation
                mode_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Unknown'
                print(f"  {col}: Imputing {nan_count_train} train + {nan_count_test} test NaN with mode '{mode_value}'")
                X_train[col] = X_train[col].fillna(mode_value)
                X_test[col] = X_test[col].fillna(mode_value)
        
        # One-Hot Encoding (low cardinality)
        ohe_cols = [col for col in categorical_cols if col != 'contry_of_res']
        
        if ohe_cols:
            print(f"\nOne-Hot Encoding: {ohe_cols}")
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(X_train[ohe_cols])
            
            X_train_ohe = pd.DataFrame(
                ohe.transform(X_train[ohe_cols]),
                columns=ohe.get_feature_names_out(ohe_cols),
                index=X_train.index
            )
            X_test_ohe = pd.DataFrame(
                ohe.transform(X_test[ohe_cols]),
                columns=ohe.get_feature_names_out(ohe_cols),
                index=X_test.index
            )
            print(f"  Created {X_train_ohe.shape[1]} one-hot encoded features")
        else:
            X_train_ohe = pd.DataFrame(index=X_train.index)
            X_test_ohe = pd.DataFrame(index=X_test.index)
        
        # Target Encoding (high cardinality - country)
        if 'contry_of_res' in categorical_cols:
            print(f"\nTarget Encoding: contry_of_res")
            te = TargetEncoder(cols=['contry_of_res'], smoothing=1.0)
            te.fit(X_train[['contry_of_res']], y_train)
            
            X_train_te = te.transform(X_train[['contry_of_res']])
            X_test_te = te.transform(X_test[['contry_of_res']])
            print(f"  Encoded country feature")
        else:
            X_train_te = pd.DataFrame(index=X_train.index)
            X_test_te = pd.DataFrame(index=X_test.index)
        
        # Handle missing values in numerical columns (fit on train, transform both)
        if numerical_cols:
            imputer = SimpleImputer(strategy='median')
            X_train_num_imputed = pd.DataFrame(
                imputer.fit_transform(X_train[numerical_cols]),
                columns=numerical_cols
            )
            X_test_num_imputed = pd.DataFrame(
                imputer.transform(X_test[numerical_cols]),
                columns=numerical_cols
            )
            
            # Check if any missing values were imputed
            missing_count = X_train[numerical_cols].isnull().sum().sum()
            if missing_count > 0:
                print(f"\nMissing value imputation:")
                print(f"  Imputed {missing_count} missing values in training data")
        else:
            X_train_num_imputed = pd.DataFrame()
            X_test_num_imputed = pd.DataFrame()
        
        # Combine all features
        X_train_processed = pd.concat([
            X_train_num_imputed.reset_index(drop=True),
            X_train_ohe.reset_index(drop=True),
            X_train_te.reset_index(drop=True)
        ], axis=1)
        
        X_test_processed = pd.concat([
            X_test_num_imputed.reset_index(drop=True),
            X_test_ohe.reset_index(drop=True),
            X_test_te.reset_index(drop=True)
        ], axis=1)
        
        print(f"\nFinal feature count: {X_train_processed.shape[1]}")
        
        # Return data AND metadata
        metadata = {
            'numerical_cols': numerical_cols,
            'ohe_cols': ohe_cols,
            'te_cols': ['contry_of_res'] if 'contry_of_res' in categorical_cols else [],
            'imputer': imputer if 'imputer' in locals() else None,
            'ohe': ohe if 'ohe' in locals() else None,
            'te': te if 'te' in locals() else None
        }
        
        return X_train_processed, X_test_processed, y_train.values, y_test.values, metadata
    
    def apply_scaling(self, X_train, X_test, scaler_name):
        """Apply scaling (fit on train, transform both)"""
        scaler = self.scalers[scaler_name]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to balance training data"""
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"\nSMOTE applied:")
        print(f"  Before: {X_train.shape[0]} samples")
        print(f"  After: {X_train_balanced.shape[0]} samples")
        print(f"  Class 0: {(y_train_balanced == 0).sum()}")
        print(f"  Class 1: {(y_train_balanced == 1).sum()}")
        
        return X_train_balanced, y_train_balanced
    
    def tune_and_train(self, model_name, X_train, y_train):
        """Hyperparameter tuning and training"""
        model = self.models[model_name]
        params = self.param_grids[model_name]
        
        # Determine tuning settings
        cv_folds = 3 if getattr(self, 'quick_mode', False) else 5
        n_iter_moderate = 5 if getattr(self, 'quick_mode', False) else 30
        n_iter_complex = 10 if getattr(self, 'quick_mode', False) else 50
        
        # Determine tuning method based on model complexity
        if model_name in ['DecisionTree', 'KNN', 'LDA', 'GaussianNB']:
            # Grid Search for simple models
            search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            tuning_method = 'GridSearch'
            
        elif model_name in ['LogisticRegression', 'AdaBoost', 'RandomForest']:
            # Bayesian Optimization for moderate models
            search = BayesSearchCV(
                estimator=model,
                search_spaces=params,
                n_iter=n_iter_moderate,
                cv=cv_folds,
                scoring='roc_auc',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            tuning_method = 'BayesianOptimization'
            
        else:  # SVM
            # Bayesian Optimization for complex model
            search = BayesSearchCV(
                estimator=model,
                search_spaces=params,
                n_iter=n_iter_complex,
                cv=cv_folds,
                scoring='roc_auc',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            tuning_method = 'BayesianOptimization'
        
        # Fit
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, tuning_method
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all 8 evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'F1-Score': f1_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Kappa': cohen_kappa_score(y_true, y_pred),
            'Log Loss': log_loss(y_true, y_pred_proba)
        }
        return metrics
    
    def run_pipeline(self, filepath, output_dir):
        """Run complete pipeline"""
        # Apply mitigation adjustments if requested
        if getattr(self, 'apply_mitigation', False):
            print('Applying overfitting mitigation: tightening hyperparameter grids')
            for model_name, params in self.param_grids.items():
                if model_name == 'DecisionTree':
                    params['max_depth'] = [3, 5, 7, 10]
                if model_name == 'RandomForest':
                    params['n_estimators'] = Integer(50, 200)
                if model_name == 'AdaBoost':
                    params['n_estimators'] = Integer(50, 200)
                if model_name == 'LogisticRegression':
                    params['C'] = Real(0.01, 5, prior='log-uniform')
        
        print(f"\n{'#'*80}")
        print(f"# AUTISM ML PIPELINE - {self.dataset_name.upper()} DATASET")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess data
        df = self.load_data(filepath)
        X_train, X_test, y_train, y_test, metadata = self.preprocess_data(df)

        # Track best model
        best_score = -1
        best_artifacts = None

        # Iterate through scalers
        for scaler_name in self.scalers.keys():
            # Apply scaling
            X_train_scaled, X_test_scaled = self.apply_scaling(
                X_train, X_test, scaler_name
            )
            # Apply SMOTE
            X_train_balanced, y_train_balanced = self.apply_smote(
                X_train_scaled, y_train
            )
            # Iterate through models
            for model_name in self.models.keys():
                try:
                    # Tune and train
                    best_model, best_params, tuning_method = self.tune_and_train(
                        model_name, X_train_balanced, y_train_balanced
                    )
                    # Predict
                    y_pred = best_model.predict(X_test_scaled)
                    y_pred_proba = best_model.predict_proba(X_test_scaled)
                    # Calculate metrics for test set
                    metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                    # Optionally calculate training metrics
                    train_metrics = {}
                    if getattr(self, 'evaluate_train', False):
                        y_train_pred = best_model.predict(X_train_balanced)
                        y_train_proba = best_model.predict_proba(X_train_balanced)
                        train_metrics = self.calculate_metrics(y_train_balanced, y_train_pred, y_train_proba)
                        train_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
                    # Store results
                    result = {
                        'dataset': self.dataset_name,
                        'scaler': scaler_name,
                        'model': model_name,
                        'tuning_method': tuning_method,
                        'best_params': str(best_params),
                        **metrics,
                        **train_metrics
                    }
                    self.results.append(result)
                    # Check if this is the best model
                    if metrics['ROC-AUC'] > best_score:
                        print(f"DEBUG: Updating best model! New score: {metrics['ROC-AUC']}")
                        best_score = metrics['ROC-AUC']
                        best_artifacts = {
                            'model': best_model,
                            'scaler': self.scalers[scaler_name],
                            'imputer': metadata['imputer'],
                            'ohe': metadata['ohe'],
                            'te': metadata['te'],
                            'numerical_cols': metadata['numerical_cols'],
                            'ohe_cols': metadata['ohe_cols'],
                            'te_cols': metadata['te_cols'],
                            'dataset_name': self.dataset_name,
                            'score': best_score,
                            'model_name': model_name,
                            'scaler_name': scaler_name
                        }
                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    continue
        
        # Save results
        self.save_results(output_dir)
        
        # Save best model
        if best_artifacts:
            model_path = os.path.join(output_dir, f'{self.dataset_name}_best_model.pkl')
            joblib.dump(best_artifacts, model_path)
            print(f"\n{'='*80}")
            print(f"BEST MODEL SAVED")
            print(f"{'='*80}")
            print(f"Model: {best_artifacts['model_name']}")
            print(f"Scaler: {best_artifacts['scaler_name']}")
            print(f"ROC-AUC: {best_artifacts['score']:.4f}")
            print(f"Saved to: {model_path}")
        
        print(f"\n{'#'*80}")
        print(f"# PIPELINE COMPLETE")
        print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Results saved to: {output_dir}")
        print(f"{'#'*80}")
    
    def save_results(self, output_dir):
        """Save results to CSV and JSON, detect overfitting, and print summary"""
        results_df = pd.DataFrame(self.results)
        # Save CSV
        csv_path = os.path.join(output_dir, f'{self.dataset_name}_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        # Save JSON
        json_path = os.path.join(output_dir, f'{self.dataset_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {json_path}")
        # Overfitting detection
        if any(col.startswith('train_') for col in results_df.columns):
            overfit_cases = []
            for _, row in results_df.iterrows():
                train_acc = row.get('train_Accuracy')
                test_acc = row.get('Accuracy')
                if train_acc == 1.0 and test_acc < 1.0:
                    overfit_cases.append((row['model'], row['scaler']))
            if overfit_cases:
                print('\n[!WARNING] Potential overfitting detected in the following model/scaler combos:')
                for model, scaler in overfit_cases:
                    print(f'  - Model: {model}, Scaler: {scaler}')
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total experiments: {len(self.results)}")
        print(f"\nBest model by ROC-AUC:")
        best_idx = results_df['ROC-AUC'].idxmax()
        best = results_df.loc[best_idx]
        print(f"  Scaler: {best['scaler']}")
        print(f"  Model: {best['model']}")
        print(f"  ROC-AUC: {best['ROC-AUC']:.4f}")
        print(f"  Accuracy: {best['Accuracy']:.4f}")
        print(f"  F1-Score: {best['F1-Score']:.4f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Autism ML Pipeline')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['adolescent', 'adult', 'child', 'toddler'],
                       help='Dataset name')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to preprocessed CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--evaluate_train', action='store_true',
                       help='Compute and log training set metrics')
    parser.add_argument('--apply_mitigation', action='store_true',
                       help='Apply overfitting mitigation by tightening hyperparameter grids')
    
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (fewer iterations, fewer CV folds)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = AutismMLPipeline(args.dataset, args.random_state)
    # Store flags for later use
    pipeline.evaluate_train = getattr(args, 'evaluate_train', False)
    pipeline.apply_mitigation = getattr(args, 'apply_mitigation', False)
    pipeline.quick_mode = getattr(args, 'quick', False)
    pipeline.run_pipeline(args.input, args.output_dir)


if __name__ == '__main__':
    main()
