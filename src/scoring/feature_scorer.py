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
from machine_learning.classification import train_model

# def score_features(X: pd.DataFrame, 
#                   y: pd.Series, 
#                   model_name: str) -> Dict[str, Dict[str, float]]:
#     """
#     Calculate performance scores for each feature.
    
#     Args:
#         X: Feature matrix
#         y: Target labels
#         model_name: Name of the model to train (e.g., 'random_forest', 'xgboost')
        
#     Returns:
#         Dictionary mapping feature names to their performance metrics
#     """
#     feature_scores = {}
    
#     for feature in X.columns:
#         # Use single feature to get predictions
#         X_feature = X[[feature]]
        
#         # Train the model using the specified model name
#         model = train_model(X_feature, y, model_name)
        
#         # Get feature importance or coefficient if available
#         feature_importance = getattr(model, 'feature_importances_', None)
#         if feature_importance is not None:
#             importance = feature_importance[0]  # Since we are using one feature
#         else:
#             importance = 0.0
            
#         y_pred = model.predict(X_feature)
        
#         # Calculate metrics
#         metrics = calculate_metrics(y, y_pred)
#         metrics["importance"] = importance
#         feature_scores[feature] = metrics
        
#     return feature_scores