"""
Module for scoring individual features in the GSM pipeline.

Functions:
- score_features: Calculate performance metrics for individual features
- calculate_feature_importance: Extract feature importance from model
"""

from typing import Dict
import numpy as np
import pandas as pd
from .metrics import calculate_metrics

def score_features(X: pd.DataFrame, 
                  y: pd.Series, 
                  model: object) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance scores for each feature.
    
    Args:
        X: Feature matrix
        y: Target labels
        model: Trained model object
        
    Returns:
        Dictionary mapping feature names to their performance metrics
    """
    feature_scores = {}
    
    for feature in X.columns:
        # Get feature importance or coefficient if available
        feature_importance = getattr(model, 'feature_importances_', None)
        if feature_importance is not None:
            feature_idx = list(X.columns).index(feature)
            importance = feature_importance[feature_idx]
        else:
            importance = 0.0
            
        # Use single feature to get predictions
        X_feature = X[[feature]]
        y_pred = model.predict(X_feature)
        
        # Calculate metrics
        metrics = calculate_metrics(y, y_pred)
        metrics["importance"] = importance
        feature_scores[feature] = metrics
        
    return feature_scores