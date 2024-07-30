# experiment_tracking.py

import mlflow
# import numpy as np
# import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

# Start MLflow experiment
mlflow.start_run()

# Logistic Regression Experiment
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Log parameters and metrics
mlflow.log_param("model_type1", "Logistic Regression")
mlflow.log_param("max_iter", 200)
mlflow.log_metric("accuracy", accuracy)

# Random Forest Experiment
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Log Random Forest experiment
mlflow.log_param("model_type2", "Random Forest")
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("rf_accuracy", rf_accuracy)

mlflow.end_run()

print(f"Logistic Regression Accuracy: {accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
