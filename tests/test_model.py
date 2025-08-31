import os
import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model/iris_model.pkl"

class TestModelTraining(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertTrue(os.path.exists(MODEL_PATH))

    def test_model_training(self):
        model = joblib.load(MODEL_PATH)
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertTrue(hasattr(model, "feature_importances_"))
        self.assertGreaterEqual(len(model.feature_importances_), 4)

if __name__ == '__main__':
    unittest.main()
