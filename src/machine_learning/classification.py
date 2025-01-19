"""
File Purpose:
This module contains functions for training and predicting with machine learning models.

Key Functions:
- train_model: Trains a specified machine learning model on the provided data.
- predict: Generates predictions using the trained model.

Usage Example:
    from machine_learning.classification import train_model, predict

    model, X_train, y_train, X_test, y_test = train_model('RandomForest', data_x, data_y)
    predictions = predict(model, X_test)
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def train_model(model: str, data_x: pd.DataFrame, data_y: pd.Series) -> Tuple:
    """
    Train a specified machine learning model on the provided data.

    Args:
        model: The name of the machine learning model to use ('RandomForest' or 'GradientBoosting').
        data_x: Feature DataFrame.
        data_y: Target Series.

    Returns:
        Tuple containing:
        - Trained model
        - Training features
        - Training target
        - Testing features
        - Testing target
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

    # Initialize the model
    if model == 'RandomForest':
        clf = RandomForestClassifier()
    elif model == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    else:
        raise ValueError(f"Model '{model}' is not supported.")

    # Train the model
    clf.fit(X_train, y_train)

    return clf, X_train, y_train, X_test, y_test

def predict(model, X_test: pd.DataFrame):
    """
    Generate predictions using the trained model.

    Args:
        model: The trained machine learning model.
        X_test: Testing features DataFrame.

    Returns:
        Predictions for the test set.
    """
    return model.predict(X_test)
