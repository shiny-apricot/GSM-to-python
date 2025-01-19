"""
Module for training machine learning models with standardized preprocessing and evaluation.

Key components:
- TrainingData: Dataclass containing training and test data splits
- ModelEvaluation: Dataclass containing model evaluation metrics
- ModelTrainer: Core functions for model training pipeline
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from machine_learning.classification import train_model as classify_train_model, predict

from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


@dataclass
class TrainingData:
    """Container for training and test data splits"""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    
    @property
    def training_size(self) -> int:
        return len(self.X_train)
    
    @property
    def test_size(self) -> int:
        return len(self.X_test)

@dataclass
class ModelEvaluation:
    """Container for model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }

def _get_scaler(normalization_method: str):
    """Get the appropriate scaler based on configuration"""
    scalers = {
        'minmax': MinMaxScaler(),
        'zscore': StandardScaler(),
        'robust': RobustScaler()
    }
    return scalers.get(normalization_method, StandardScaler())

def prepare_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int
) -> TrainingData:
    """
    Prepare and split the dataset into training and testing sets.
    
    Args:
        data: Input DataFrame containing features and target
        target_column: Name of the target variable column
        test_size: Proportion of dataset to include in the test split
        random_state: Random state for reproducibility
    
    Returns:
        TrainingData object containing the splits
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - test_size,  # Convert train ratio to test ratio
        random_state=random_state,
        stratify=y if len(y.unique()) < 10 else None
    )
    
    return TrainingData(X_train, X_test, y_train, y_test)

def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    NORMALIZATION_METHOD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess features using configured normalization method.
    
    Args:
        X_train: Training features
        X_test: Testing features
    
    Returns:
        Tuple of preprocessed (X_train, X_test)
    """
    scaler = _get_scaler(NORMALIZATION_METHOD)
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled

def train_model(
    model: BaseEstimator,
    training_data: TrainingData,
    HYPERPARAMETERS
) -> BaseEstimator:
    """
    Train a machine learning model with the given data.
    
    Args:
        model: Scikit-learn compatible model
        training_data: TrainingData object containing splits
        
    Returns:
        Trained model
    """
    # Configure model hyperparameters if available
    if hasattr(model, 'get_params'):
        model_name = model.__class__.__name__
        if model_name in HYPERPARAMETERS:
            model.set_params(**HYPERPARAMETERS[model_name])
    
    # Use the classification module's train_model function
    return classify_train_model(model, training_data)

def evaluate_model(
    model: BaseEstimator,
    training_data: TrainingData
) -> ModelEvaluation:
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained model
        training_data: TrainingData object containing test data
    
    Returns:
        ModelEvaluation object containing metrics
    """
    y_pred = model.predict(training_data.X_test)
    
    # Use weighted average for multi-class problems, binary for binary classification
    average = 'weighted' if len(np.unique(training_data.y_test)) > 2 else 'binary'
    
    return ModelEvaluation(
        accuracy=accuracy_score(training_data.y_test, y_pred),
        precision=precision_score(training_data.y_test, y_pred, average=average),
        recall=recall_score(training_data.y_test, y_pred, average=average),
        f1=f1_score(training_data.y_test, y_pred, average=average)
    )

def run_training_pipeline(
    model: BaseEstimator,
    data: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    HYPERPARAMETERS,
    NORMALIZATION_METHOD
) -> Tuple[BaseEstimator, ModelEvaluation]:
    """
    Run the complete training pipeline including data preparation,
    preprocessing, model training, and evaluation.
    
    Args:
        model: Scikit-learn model instance
        data: Input DataFrame
        target_column: Name of target column
        test_size: Test set proportion (from config)
        random_state: Random seed (from config)
    
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> data = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [2, 4, 5, 4, 5],
        ...     'target': [0, 1, 1, 0, 1]
        ... })
        >>> trained_model, evaluation = run_training_pipeline(
        ...     model, data, 'target'
        ... )
        >>> print(evaluation.accuracy)
        0.8
    """
    # Split and preprocess data
    training_data = prepare_data(data, target_column, test_size, random_state)
    X_train_processed, X_test_processed = preprocess_features(
        training_data.X_train, 
        training_data.X_test,
        NORMALIZATION_METHOD
    )
    
    # Update training data with processed features
    training_data.X_train = X_train_processed
    training_data.X_test = X_test_processed
    
    # Train and evaluate model
    trained_model = train_model(model, training_data, HYPERPARAMETERS)
    evaluation = evaluate_model(trained_model, training_data)
    
    return trained_model, evaluation