import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add current directory to path to import ml_pipeline
sys.path.append(os.getcwd())
from ml_pipeline import AutismMLPipeline

class TestOverfittingDetection(unittest.TestCase):
    def setUp(self):
        self.pipeline = AutismMLPipeline('test_dataset')
        self.pipeline.evaluate_train = True

    def test_overfitting_detection_logic(self):
        # Simulate results where one model is overfitting
        self.pipeline.results = [
            {
                'dataset': 'test_dataset',
                'scaler': 'QuantileTransformer',
                'model': 'DecisionTree',
                'tuning_method': 'GridSearch',
                'best_params': "{'max_depth': None}",
                'Accuracy': 0.85,
                'ROC-AUC': 0.90,
                'F1-Score': 0.86,
                'train_Accuracy': 1.0,  # Overfitting!
                'train_ROC-AUC': 1.0
            },
            {
                'dataset': 'test_dataset',
                'scaler': 'QuantileTransformer',
                'model': 'LogisticRegression',
                'tuning_method': 'BayesianOptimization',
                'best_params': "{'C': 1.0}",
                'Accuracy': 0.88,
                'ROC-AUC': 0.92,
                'F1-Score': 0.89,
                'train_Accuracy': 0.90,  # Not overfitting
                'train_ROC-AUC': 0.93
            }
        ]

        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            # Create a temporary directory for output
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.pipeline.save_results(tmpdirname)
                
                # Check if warning was printed
                # We need to check if any of the print calls contained the warning message
                warning_printed = False
                for call in mock_print.call_args_list:
                    args, _ = call
                    if args and '[!WARNING] Potential overfitting detected' in str(args[0]):
                        warning_printed = True
                        break
                
                self.assertTrue(warning_printed, "Overfitting warning should be printed")
                
                # Check if specific model was mentioned
                model_mentioned = False
                for call in mock_print.call_args_list:
                    args, _ = call
                    if args and 'Model: DecisionTree' in str(args[0]):
                        model_mentioned = True
                        break
                self.assertTrue(model_mentioned, "Overfitting model should be listed")

if __name__ == '__main__':
    unittest.main()
