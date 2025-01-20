"""
Module containing metric calculation utilities for the GSM pipeline.

Functions:
- calculate_metrics: Calculate common ML evaluation metrics
- calculate_average_scores: Calculate average scores across features/groups
- rank_by_score: Rank items by their scores
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class MetricsData:
    name: Optional[str] = None

    # Basic metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    specificity: Optional[float] = None
    roc_auc: Optional[float] = None

    # Standard deviations
    accuracy_std: Optional[float] = None
    precision_std: Optional[float] = None
    recall_std: Optional[float] = None
    f1_std: Optional[float] = None
    specificity_std: Optional[float] = None
    roc_auc_std: Optional[float] = None

    # Confidence metrics
    mean_confidence: Optional[float] = None
    high_confidence_ratio: Optional[float] = None
    decision_boundary_ratio: Optional[float] = None


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsData:
    """
    Calculate common ML evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Metrics dataclass containing accuracy, precision, recall, and F1 scores
    """
    return MetricsData(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred))
    )

def calculate_average_scores(scores: Dict[str, MetricsData]) -> MetricsData:
    """
    Calculate average scores across all features or groups.
    
    Args:
        scores: Dictionary of Metrics dataclass
        
    Returns:
        Metrics dataclass of averaged metrics
    """
    return MetricsData(
        accuracy=float(np.mean([score.accuracy for score in scores.values() if score.accuracy is not None])),
        precision=float(np.mean([score.precision for score in scores.values() if score.precision is not None])),
        recall=float(np.mean([score.recall for score in scores.values() if score.recall is not None])),
        f1=float(np.mean([score.f1 for score in scores.values() if score.f1 is not None]))
    )


def rank_by_score(scores: List[MetricsData], metric: str = "f1") -> List[MetricsData]:
    """
    Rank features or groups by their scores.
    
    Args:
        scores: List of MetricsData objects
        metric: Metric to rank by (default is "f1")
        
    Returns:
        List of MetricsData objects sorted by the specified metric in descending order
    """
    valid_metrics = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric '{metric}'. Choose from {valid_metrics}.")

    # Sort the scores based on the specified metric in descending order
    return sorted(scores, key=lambda x: getattr(x, metric), reverse=True)