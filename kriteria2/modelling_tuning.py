# modelling_tuning.py

import pandas as pd
import mlflow
import mlflow.sklearn

# Set tracking URI ke MLflow server lokal di port 5000
mlflow.set_tracking_uri("http://localhost:5000")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

with mlflow.start_run():
    try:
        # Data
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )

        # Hyperparameter
        max_depth = 10
        n_estimators = 50

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        # Model
        clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)

        mlflow.sklearn.log_model(clf, "model")

    except Exception as e:
        mlflow.log_param("error", str(e))
        raise
