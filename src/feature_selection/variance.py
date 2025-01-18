"""
Variance-based feature selection methods.

This module provides functions for selecting features based on their variance:
- Constant feature removal
- Low variance feature removal
- High correlation feature removal

Key Functions:
    - remove_constant_features: Remove features with zero variance
    - remove_low_variance: Remove features with variance below threshold
    - remove_highly_correlated: Remove highly correlated features
"""

import pandas as pd
import numpy as np
from typing import List

def remove_constant_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that have zero variance (constant values).
    
    Args:
        X: Feature DataFrame
        
    Returns:
        DataFrame with constant features removed
        
    Example:
        >>> X_cleaned = remove_constant_features(X)
    """
    variance = X.var()
    non_constant_cols = variance[variance != 0].index
    return X[non_constant_cols]

def remove_low_variance(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features with variance below specified threshold.
    
    Args:
        X: Feature DataFrame
        threshold: Minimum variance threshold
        
    Returns:
        DataFrame with low variance features removed
    """
    variance = X.var()
    selected_cols = variance[variance > threshold].index
    return X[selected_cols]

def remove_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove features that are highly correlated with other features.
    Keeps one feature from each group of correlated features.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold (0.0 to 1.0)
        
    Returns:
        DataFrame with highly correlated features removed
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    features_to_drop = [
        column for column in upper.columns 
        if any(upper[column] > threshold)
    ]
    
    return X.drop(columns=features_to_drop) 