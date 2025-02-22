"""
Main scoring module for the GSM pipeline.

This module coordinates feature and group scoring operations.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from config import CROSS_VALIDATION_FOLDS
from utils.save_ranked_features import save_ranked_features, FeatureRankingOutput
from utils.save_ranked_groups import save_ranked_groups
from .score_data import ScoringParameters, score_data
from .metrics import MetricsData, rank_by_score
from .feature_scorer import score_features, FeatureScore
from grouping.group_feature_mapping import GroupFeatureMappingData


@dataclass
class ScoringResults:
    """Container for all scoring results."""
    ranked_groups: List[MetricsData]
    feature_scores: List[FeatureScore]
    group_feature_mapping: dict[str, List[str]]


def run_scoring(
    data_x: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    groups: List[GroupFeatureMappingData],
    output_dir: Path,
    iteration: int,
    logger
) -> ScoringResults:
    """
    Run the complete scoring pipeline for both groups and features.
    
    Args:
        data_x: Feature matrix
        labels: Target labels
        model_name: Name of the model to use
        groups: Group assignments
        output_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        ScoringResults containing ranked groups and feature scores
    """
    try:
        # Score individual features
        feature_scores = score_all_features(data_x, labels, logger)
        
        # Process groups
        processed_group_scores = []
        group_features = {}  # Track features per group
        
        logger.info("🔄 Starting group scoring process...")
        for current_group in tqdm(groups, desc="📊 Scoring groups"):
            available_features = [
                feature for feature in current_group.feature_list 
                if feature in data_x.columns
            ]
            
            if not available_features:
                logger.warning(f"⚠️ Skipping group with no valid features: {current_group.group_name}")
                continue
                
            group_features[current_group.group_name] = available_features
            group_data = data_x[available_features]

            scoring_params = ScoringParameters(
                data_x=group_data,
                group_name=current_group.group_name,
                labels=labels,
                classifier_name=model_name,
                cross_validation_folds=CROSS_VALIDATION_FOLDS,
                logger=logger
            )
            
            current_score = score_data(scoring_params)
            current_score.name = current_group.group_name
            processed_group_scores.append(current_score)
        
        ranked_groups = rank_by_score(
            metrics_list=processed_group_scores,
            score_type="f1",
            logger=logger
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ranked groups
        groups_output = output_dir / f"ranked_groups_iteration_{iteration}.xlsx"
        save_ranked_groups(
            str(groups_output),
            ranked_groups.metrics,
            iteration=1,  # You might want to make this configurable
            logger=logger
        )

        # Save ranked features
        features_output = output_dir / f"ranked_features_iteration_{iteration}.xlsx"
        ranking_output = FeatureRankingOutput(
            output_path=features_output,
            feature_scores=feature_scores,
            timestamp=timestamp,
            model_name=model_name
        )
        save_ranked_features(ranking_output, logger)
        
        return ScoringResults(
            ranked_groups=ranked_groups.metrics,
            feature_scores=feature_scores,
            group_feature_mapping=group_features
        )

    except Exception as e:
        logger.error(f"❌ Scoring pipeline failed: {str(e)}")
        raise


def score_all_features(
    data_x: pd.DataFrame,
    labels: pd.Series,
    logger
) -> List[FeatureScore]:
    """Score all features in the dataset."""
    logger.info("🎯 Starting feature scoring...")
    feature_scores = score_features(
        data_x=data_x,
        labels=labels,
        feature_names=list(data_x.columns),
        logger=logger
    )
    logger.info(f"✅ Completed scoring {len(feature_scores)} features")
    return feature_scores
