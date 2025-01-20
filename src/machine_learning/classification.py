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

from email.mime import base
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_model(model_name: str, data_x: pd.DataFrame, data_y: pd.Series):
    """
    Train a specified machine learning model on the provided data.

    Args:
        model: The name of the machine learning model to use ('RandomForest' or 'GradientBoosting').
        data_x: Feature DataFrame.
        data_y: Target Series.

    Returns:
        Tuple containing:
        - Trained model
    """

    # Initialize the model
    if model_name == 'RandomForest':
        classifier = RandomForestClassifier()
    elif model_name == 'GradientBoosting':
        classifier = GradientBoostingClassifier()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Train the model
    classifier.fit(data_x, data_y)

    return classifier

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

def get_classifier_object(model_name: str):
    """
    Get the classifier object based on the model name.

    Args:
        model_name: The name of the machine learning model to use ('RandomForest' or 'GradientBoosting').

    Returns:
        The classifier object.
    """
    if model_name == 'RandomForest':
        return RandomForestClassifier()
    elif model_name == 'GradientBoosting':
        return GradientBoostingClassifier()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
