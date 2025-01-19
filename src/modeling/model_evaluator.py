# model_evaluator.py

"""
Model Evaluation Module

This module provides functions for evaluating machine learning models in the GSM pipeline.
It includes functionality for:
- Model performance evaluation with comprehensive metrics
- Prediction generation
- Feature importance tracking
- Confidence score calculation

Key Functions:
- evaluate_model: Generate predictions and evaluate model performance
- calculate_metrics: Compute comprehensive classification metrics
- update_feature_ranks: Track and update feature importance rankings
- calculate_confidence_scores: Generate prediction confidence metrics
"""

from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix
)

def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger,
    return_proba: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate model performance and generate predictions.
    
    Args:
        model: Trained sklearn-compatible model
        X_test: Test features DataFrame
        y_test: Test target Series
        return_proba: Whether to return probability predictions
        
    Returns:
        Tuple of (predictions, prediction_probabilities)
        prediction_probabilities will be None if return_proba=False
        
    Raises:
        ValueError: If input data shapes don't match or model is not fitted
    """
    if not hasattr(model, "predict"):
        raise ValueError("Model must implement predict method")
        
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if return_proba else None
        
        logger.info(f"Generated predictions for {len(X_test)} samples")
        return y_pred, y_pred_proba
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    logger,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary containing metrics:
        - ROC AUC (if probabilities provided)
        - Precision
        - Recall
        - F1 Score
        - Matthews Correlation Coefficient
        - Balanced Accuracy
        - Confusion Matrix elements
    """
    try:
        metrics = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }
        
        # Add ROC AUC if probabilities are provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
        # Add confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def update_feature_ranks(
    current_ranks: pd.DataFrame,
    feature_scores: pd.Series,
    logger,
    method: str = 'mean',
    weights: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Update feature importance rankings with new scores.
    
    Args:
        current_ranks: Current feature rankings DataFrame
        feature_scores: New feature importance scores
        method: Method to combine scores ('mean', 'max', or 'weighted')
        weights: Weights for weighted averaging (required if method='weighted')
        
    Returns:
        Updated feature rankings DataFrame
        
    Raises:
        ValueError: If invalid method specified or weights missing for weighted method
    """
    if method not in ['mean', 'max', 'weighted']:
        raise ValueError("Method must be one of: 'mean', 'max', 'weighted'")
        
    if method == 'weighted' and weights is None:
        raise ValueError("Weights must be provided for weighted averaging")
        
    try:
        combined = pd.concat([current_ranks, feature_scores])
        
        if method == 'mean':
            return combined.groupby('feature').mean()
        elif method == 'max':
            return combined.groupby('feature').max()
        else:  # weighted
            return (combined.groupby('feature')
                   .apply(lambda x: np.average(x, weights=weights)))
            
    except Exception as e:
        logger.error(f"Error updating feature ranks: {str(e)}")
        raise

def calculate_confidence_scores(
    y_pred_proba: np.ndarray,
    logger,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate confidence metrics for predictions.
    
    Args:
        y_pred_proba: Prediction probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary containing:
        - mean_confidence: Average prediction confidence
        - high_confidence_ratio: Proportion of high-confidence predictions
        - decision_boundary_ratio: Proportion of predictions near decision boundary
    """
    try:
        confidence_scores = np.abs(y_pred_proba - threshold)
        high_conf_mask = confidence_scores >= 0.25
        boundary_mask = confidence_scores <= 0.1
        
        return {
            'mean_confidence': float(np.mean(confidence_scores)),
            'high_confidence_ratio': float(np.mean(high_conf_mask)),
            'decision_boundary_ratio': float(np.mean(boundary_mask))
        }
        
    except Exception as e:
        logger.error(f"Error calculating confidence scores: {str(e)}")
        raise