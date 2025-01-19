"""
Main scoring module for the GSM pipeline.

This module provides the main entry point for the scoring pipeline,
coordinating feature and group scoring operations.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .feature_scorer import score_features
from .group_scorer import score_groups
from .metrics import rank_by_score

def run_scoring(data: pd.DataFrame,
               model: str,
               groups: pd.DataFrame,
               labels: np.ndarray,
               predictions: np.ndarray) -> Tuple[List, List, Dict]:
    """
    Run the complete scoring pipeline.
    
    Args:
        data: Feature matrix
        model: Trained model
        groups: Group assignments
        labels: True labels
        predictions: Model predictions
        
    Returns:
        Tuple containing:
        - Ranked features
        - Ranked groups
        - Feature scores dictionary
    """
    feature_scores = score_features(data, labels, model)
    group_scores = score_groups(groups, predictions, labels)
    
    ranked_features = rank_by_score(feature_scores)
    ranked_groups = rank_by_score(group_scores)
    
    return ranked_features, ranked_groups, feature_scores

