"""
Feature selection using SelectKBest methods.

This module provides functions for selecting top features using different statistical tests:
- Chi-squared test (for classification)
- F-test (for regression)
- Mutual information (works for both classification and regression)

Key Functions:
    - select_k_best_chi2: Select features using chi-squared test
    - select_k_best_f: Select features using F-test
    - select_k_best_mutual_info: Select features using mutual information
"""

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import pandas as pd
import numpy as np
from typing import Tuple

def select_k_best_chi2(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Select top k features using chi-squared test (for classification tasks).
    
    Args:
        X: Feature DataFrame with non-negative values
        y: Target variable
        k: Number of features to select
        
    Returns:
        Tuple containing:
        - Selected feature DataFrame
        - Array of feature scores
        
    Example:
        >>> X_selected, scores = select_k_best_chi2(X, y, k=10)
    """
    # Ensure non-negative values for chi-squared test
    if (X < 0).any().any():
        raise ValueError("Chi-squared test requires non-negative values")
        
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)
    
    return X_selected, selector.scores_

def select_k_best_f(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Select top k features using F-test (works for both classification and regression).
    
    Args:
        X: Feature DataFrame
        y: Target variable
        k: Number of features to select
        
    Returns:
        Tuple containing:
        - Selected feature DataFrame
        - Array of feature scores
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)
    
    return X_selected, selector.scores_

def select_k_best_mutual_info(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Select top k features using mutual information (works for both classification and regression).
    
    Args:
        X: Feature DataFrame
        y: Target variable
        k: Number of features to select
        
    Returns:
        Tuple containing:
        - Selected feature DataFrame
        - Array of feature scores
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)
    
    return X_selected, selector.scores_