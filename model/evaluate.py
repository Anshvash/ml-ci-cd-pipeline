import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
data_path = os.path.join(BASE_DIR, 'data', 'iris.csv')

# Load the dataset
data = pd.read_csv(data_path)

# Preprocess the dataset
X = data.drop('species', axis=1)
y = data['species']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
data_path = os.path.join(BASE_DIR, 'iris_model.pkl')
model = joblib.load(data_path)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
