"""
Module containing metric calculation utilities for the GSM pipeline.

Functions:
- calculate_metrics: Calculate common ML evaluation metrics
- calculate_average_scores: Calculate average scores across features/groups
- rank_by_score: Rank items by their scores
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common ML evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 scores
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def calculate_average_scores(scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate average scores across all features or groups.
    
    Args:
        scores: Dictionary of score metrics
        
    Returns:
        Dictionary of averaged metrics
    """
    metrics = list(next(iter(scores.values())).keys())
    return {
        metric: np.mean([score[metric] for score in scores.values()])
        for metric in metrics
    }

def rank_by_score(scores: Dict[str, Dict[str, float]], 
                 metric: str = "f1") -> List[Tuple[str, float]]:
    """
    Rank features or groups by their scores.
    
    Args:
        scores: Dictionary of score metrics
        metric: Metric to use for ranking (default: f1)
        
    Returns:
        List of (name, score) tuples sorted by score
    """
    return sorted(
        [(name, metrics[metric]) for name, metrics in scores.items()],
        key=lambda x: x[1],
        reverse=True
    )