"""
Modified ML Pipeline for Toddler Dataset - WITHOUT A1-A10 Features

This script trains a model using ONLY demographic features:
- age_months
- gender
- ethnicity
- jaundice
- family_asd
- relation

The goal is to predict autism risk based on demographics alone, not behavioral questions.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer, MaxAbsScaler, OneHotEncoder
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
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, cohen_kappa_score, log_loss
)

class ToddlerDemographicPipeline:
    """ML Pipeline for Toddler Dataset using ONLY demographic features"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        
        # Define scalers
        self.scalers = {
            'QuantileTransformer': QuantileTransformer(output_distribution='normal', random_state=random_state),
            'PowerTransformer': PowerTransformer(method='yeo-johnson', standardize=True),
            'Normalizer': Normalizer(norm='l2'),
            'MaxAbsScaler': MaxAbsScaler()
        }
        
        # Define models
        self.models = self._get_models()
        self.param_grids = self._get_param_grids()
        
    def _get_models(self):
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
        return {
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
            'SVM': {
                'C': Real(0.1, 100, prior='log-uniform'),
                'gamma': Real(1e-4, 1e-1, prior='log-uniform'),
            }
        }
    
    def load_data(self, filepath):
        """Load preprocessed dataset"""
        print(f"\n{'='*80}")
        print(f"Loading toddler dataset from: {filepath}")
        print(f"{'='*80}")
        
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess data - DROP A1-A10, use only demographics"""
        print(f"\n{'='*80}")
        print("PREPROCESSING - DEMOGRAPHICS ONLY (NO A1-A10)")
        print(f"{'='*80}")
        
        # Separate features and target
        target_col = 'Class/ASD'
        
        # DROP A1-A10 scores (this is the key change!)
        a_cols = [f'A{i}' for i in range(1, 11)]
        drop_cols = [target_col] + a_cols
        
        print(f"Dropping behavioral columns: {a_cols}")
        
        X = df.drop(drop_cols, axis=1)
        y = (df[target_col] == 'YES').astype(int)
        
        print(f"\nFeatures used: {list(X.columns)}")
        print(f"\nTarget distribution:")
        print(f"  Class 0 (NO): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
        print(f"  Class 1 (YES): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain/Test split (80/20):")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        # Identify feature types
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\nFeature types:")
        print(f"  Categorical: {len(categorical_cols)} - {categorical_cols}")
        print(f"  Numerical: {len(numerical_cols)} - {numerical_cols}")
        
        # One-Hot Encoding (for low cardinality)
        ohe_cols = [col for col in categorical_cols if col not in ['ethnicity']]
        
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
        
        # Target Encoding (for high cardinality - ethnicity)
        if 'ethnicity' in categorical_cols:
            print(f"\nTarget Encoding: ethnicity")
            te = TargetEncoder(cols=['ethnicity'], smoothing=1.0)
            te.fit(X_train[['ethnicity']], y_train)
            
            X_train_te = te.transform(X_train[['ethnicity']])
            X_test_te = te.transform(X_test[['ethnicity']])
            print(f"  Encoded ethnicity feature")
        else:
            X_train_te = pd.DataFrame(index=X_train.index)
            X_test_te = pd.DataFrame(index=X_test.index)
        
        # Handle numerical columns
        if numerical_cols:
            imputer = SimpleImputer(strategy='median')
            X_train_num = pd.DataFrame(
                imputer.fit_transform(X_train[numerical_cols]),
                columns=numerical_cols
            )
            X_test_num = pd.DataFrame(
                imputer.transform(X_test[numerical_cols]),
                columns=numerical_cols
            )
        else:
            X_train_num = pd.DataFrame()
            X_test_num = pd.DataFrame()
            imputer = None
        
        # Combine all features
        X_train_processed = pd.concat([
            X_train_num.reset_index(drop=True),
            X_train_ohe.reset_index(drop=True),
            X_train_te.reset_index(drop=True)
        ], axis=1)
        
        X_test_processed = pd.concat([
            X_test_num.reset_index(drop=True),
            X_test_ohe.reset_index(drop=True),
            X_test_te.reset_index(drop=True)
        ], axis=1)
        
        print(f"\nFinal feature count: {X_train_processed.shape[1]}")
        
        # Return data AND metadata
        metadata = {
            'numerical_cols': numerical_cols,
            'ohe_cols': ohe_cols,
            'te_cols': ['ethnicity'] if 'ethnicity' in categorical_cols else [],
            'imputer': imputer,
            'ohe': ohe if 'ohe' in locals() else None,
            'te': te if 'te' in locals() else None
        }
        
        return X_train_processed, X_test_processed, y_train.values, y_test.values, metadata
    
    def apply_scaling(self, X_train, X_test, scaler_name):
        """Apply scaling"""
        scaler = self.scalers[scaler_name]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE"""
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
        
        # Quick mode settings
        cv_folds = 3
        n_iter = 10
        
        if model_name in ['DecisionTree', 'KNN', 'LDA', 'GaussianNB']:
            search = GridSearchCV(
                estimator=model, param_grid=params, cv=cv_folds,
                scoring='roc_auc', n_jobs=-1, verbose=0
            )
            tuning_method = 'GridSearch'
        else:
            search = BayesSearchCV(
                estimator=model, search_spaces=params, n_iter=n_iter,
                cv=cv_folds, scoring='roc_auc', random_state=self.random_state,
                n_jobs=-1, verbose=0
            )
            tuning_method = 'BayesianOptimization'
        
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, tuning_method
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
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
        print(f"\n{'#'*80}")
        print(f"# TODDLER DEMOGRAPHIC-ONLY PIPELINE")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess
        df = self.load_data(filepath)
        X_train, X_test, y_train, y_test, metadata = self.preprocess_data(df)
        
        # Track best model
        best_score = -1
        best_artifacts = None
        
        # Iterate through scalers
        for scaler_name in self.scalers.keys():
            X_train_scaled, X_test_scaled = self.apply_scaling(X_train, X_test, scaler_name)
            X_train_balanced, y_train_balanced = self.apply_smote(X_train_scaled, y_train)
            
            # Iterate through models
            for model_name in self.models.keys():
                try:
                    best_model, best_params, tuning_method = self.tune_and_train(
                        model_name, X_train_balanced, y_train_balanced
                    )
                    
                    y_pred = best_model.predict(X_test_scaled)
                    y_pred_proba = best_model.predict_proba(X_test_scaled)
                    metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                    
                    result = {
                        'dataset': 'toddler_demographic',
                        'scaler': scaler_name,
                        'model': model_name,
                        'tuning_method': tuning_method,
                        'best_params': str(best_params),
                        **metrics
                    }
                    self.results.append(result)
                    
                    if metrics['ROC-AUC'] > best_score:
                        print(f"New best model: {model_name} with {scaler_name} - ROC-AUC: {metrics['ROC-AUC']:.4f}")
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
                            'dataset_name': 'toddler',
                            'score': best_score,
                            'model_name': model_name,
                            'scaler_name': scaler_name,
                            'features_used': 'demographic_only_no_A1_A10'
                        }
                except Exception as e:
                    print(f"  ERROR with {model_name}: {str(e)}")
                    continue
        
        # Save results
        results_df = pd.DataFrame(self.results)
        csv_path = os.path.join(output_dir, 'toddler_demographic_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        json_path = os.path.join(output_dir, 'toddler_demographic_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # Save best model
        if best_artifacts:
            model_path = os.path.join(output_dir, 'toddler_best_model.pkl')
            joblib.dump(best_artifacts, model_path)
            print(f"\n{'='*80}")
            print(f"BEST MODEL SAVED")
            print(f"{'='*80}")
            print(f"Model: {best_artifacts['model_name']}")
            print(f"Scaler: {best_artifacts['scaler_name']}")
            print(f"ROC-AUC: {best_artifacts['score']:.4f}")
            print(f"Features: Demographic only (NO A1-A10)")
            print(f"Saved to: {model_path}")
        
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
        
        print(f"\n{'#'*80}")
        print(f"# PIPELINE COMPLETE")
        print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")

if __name__ == '__main__':
    pipeline = ToddlerDemographicPipeline(random_state=42)
    pipeline.run_pipeline(
        filepath='Autism_Toddler_Data_Preprocessed.csv',
        output_dir='.'
    )
