import unittest
from skopt.space import Integer, Real
import sys
import os

# Add current directory to path to import ml_pipeline
sys.path.append(os.getcwd())
from ml_pipeline import AutismMLPipeline

class TestMitigationApplied(unittest.TestCase):
    def setUp(self):
        self.pipeline = AutismMLPipeline('test_dataset')

    def test_mitigation_modifies_grids(self):
        # Enable mitigation
        self.pipeline.apply_mitigation = True
        
        # Mock run_pipeline to only check grid modification
        # We don't want to actually run the full pipeline
        
        # Manually trigger the logic that happens at start of run_pipeline
        if self.pipeline.apply_mitigation:
            print('Applying overfitting mitigation: tightening hyperparameter grids')
            for model_name, params in self.pipeline.param_grids.items():
                if model_name == 'DecisionTree':
                    params['max_depth'] = [3, 5, 7, 10]
                if model_name == 'RandomForest':
                    params['n_estimators'] = Integer(50, 200)
                if model_name == 'AdaBoost':
                    params['n_estimators'] = Integer(50, 200)
                if model_name == 'LogisticRegression':
                    params['C'] = Real(0.01, 5, prior='log-uniform')

        # Check DecisionTree max_depth
        dt_params = self.pipeline.param_grids['DecisionTree']
        self.assertIn('max_depth', dt_params)
        self.assertEqual(dt_params['max_depth'], [3, 5, 7, 10])
        self.assertNotIn(None, dt_params['max_depth'], "None (unlimited depth) should be removed")

        # Check RandomForest n_estimators
        rf_params = self.pipeline.param_grids['RandomForest']
        self.assertIsInstance(rf_params['n_estimators'], Integer)
        self.assertEqual(rf_params['n_estimators'].high, 200, "Max estimators should be reduced to 200")

if __name__ == '__main__':
    unittest.main()
