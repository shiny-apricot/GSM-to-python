from typing import Tuple, Dict
import pandas as pd
from sklearn.base import BaseEstimator

from .model_trainer import train_model
from .model_evaluator import evaluate_model, calculate_metrics, update_feature_ranks
from .modeling_utils import preprocess_features, filter_best_groups
from machine_learning.classification import train_model, predict


def run_modeling(
    data_x: pd.DataFrame,
    data_y: pd.Series,
    model_name: str,
    feature_ranks: pd.DataFrame = None,
    group_ranks: pd.DataFrame = None
) -> Tuple[BaseEstimator, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Run the complete modeling pipeline.
    
    Args:
        data_x: Feature DataFrame
        data_y: Target Series
        model: Machine learning model to use
        feature_ranks: Current feature rankings (optional)
        group_ranks: Current group rankings (optional)
        
    Returns:
        Tuple containing:
        - Trained model
        - Performance metrics
        - Updated feature rankings
        - Updated group rankings
    """
    # Preprocess features
    processed_x = preprocess_features(data_x.copy())
    
    # Train model
    trained_model, X_train, y_train, X_test, y_test = train_model(
        model=model_name,
        data_x=processed_x,
        data_y=data_y
    )
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(trained_model, X_test, y_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Update rankings if provided
    if feature_ranks is not None:
        feature_importance = pd.Series(
            trained_model.feature_importances_,
            index=processed_x.columns,
            name='importance'
        )
        feature_ranks = update_feature_ranks(feature_ranks, feature_importance)
    
    if group_ranks is not None:
        group_ranks = update_feature_ranks(group_ranks, pd.Series(metrics))
    
    return trained_model, metrics, feature_ranks, group_ranks
