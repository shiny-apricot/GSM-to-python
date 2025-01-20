"""
Main scoring module for the GSM pipeline.

This module provides the main entry point for the scoring pipeline,
coordinating feature and group scoring operations.
"""

from typing import Dict, List, Tuple
from venv import logger
import numpy as np
import pandas as pd

from .feature_scorer import score_features
from .data_scorer import score_data
from .metrics import MetricsData, rank_by_score
from grouping.group_feature_mapping import GroupFeatureMappingData
from IPython.display import display
from tqdm import tqdm

def run_scoring(data_x: pd.DataFrame,
                labels: pd.Series,
                model_name: str,
                logger,
                groups: List[GroupFeatureMappingData]) -> List[MetricsData]:
    """
    Run the complete scoring pipeline.
    
    Args:
        data_x: Feature matrix
        labels: True labels
        model_name: Name of the model to use for scoring
        groups: Group assignments
        
    Returns:
        Tuple containing:
        - Ranked features
        - Ranked groups
        - Feature scores dictionary
    """
    processed_group_scores = []
    logger.info("🔄 Starting group scoring process...")
    
    for current_group in tqdm(groups,
                            desc="📊 Scoring groups",
                            unit="group",
                            ncols=80,
                            position=0,
                            leave=True):
        available_features = [feature for feature in current_group.feature_list 
                            if feature in data_x.columns]
        
        if not available_features:
            # logger.warning(f"⚠️ No valid features found for group: {current_group.group_name}")
            continue
            
        group_feature_data = data_x[available_features]
        current_score = score_data(group_feature_data, labels, classifier_name=model_name)
        current_score.name = current_group.group_name
        processed_group_scores.append(current_score)
    
    ranked_groups = rank_by_score(processed_group_scores, metric="f1")
    
    return ranked_groups

