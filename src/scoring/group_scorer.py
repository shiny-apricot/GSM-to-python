"""
Module for scoring gene groups in the GSM pipeline.

Functions:
- score_groups: Calculate performance metrics for gene groups
"""

from typing import Dict
import numpy as np
import pandas as pd
from .metrics import calculate_metrics

def score_groups(groups: pd.DataFrame,
                predictions: np.ndarray,
                labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance scores for each group.
    
    Args:
        groups: DataFrame containing group assignments
        predictions: Model predictions
        labels: True labels
        
    Returns:
        Dictionary mapping group names to their performance metrics
    """
    group_scores = {}
    
    for group_name in groups["group_name"].unique():
        group_mask = groups["group_name"] == group_name
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        group_scores[group_name] = calculate_metrics(group_labels, group_preds)
    
    return group_scores