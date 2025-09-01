import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
data_path = os.path.join(BASE_DIR, 'iris_model.pkl')
model = joblib.load(data_path)

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        # model = joblib.load('iris_model.pkl')
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertGreaterEqual(len(model.feature_importances_), 4)

if __name__ == '__main__':
    unittest.main()
