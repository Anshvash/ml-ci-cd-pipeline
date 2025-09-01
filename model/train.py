import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')
