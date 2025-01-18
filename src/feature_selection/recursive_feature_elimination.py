"""
Recursive feature elimination methods.

This module implements recursive feature elimination (RFE) approaches:
- Standard RFE
- RFE with cross-validation
- Stability-based RFE

Key Functions:
    - recursive_feature_elimination: Standard RFE
    - rfe_with_cv: RFE with cross-validation
"""

from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Tuple, Union
from sklearn.base import BaseEstimator

def recursive_feature_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10,
    estimator: BaseEstimator = None
) -> Tuple[pd.DataFrame, list]:
    """
    Perform recursive feature elimination to select features.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        n_features: Number of features to select
        estimator: Sklearn estimator (defaults to RandomForestClassifier)
        
    Returns:
        Tuple containing:
        - Selected feature DataFrame
        - List of feature rankings
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
    selector = RFE(estimator=estimator, n_features_to_select=n_features)
    X_new = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.support_]
    X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)
    
    return X_selected, selector.ranking_

def rfe_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    min_features: int = 5,
    estimator: BaseEstimator = None,
    cv: int = 5
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform recursive feature elimination with cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        min_features: Minimum number of features to select
        estimator: Sklearn estimator (defaults to RandomForestClassifier)
        cv: Number of cross-validation folds
        
    Returns:
        Tuple containing:
        - Selected feature DataFrame
        - Dict with CV results
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
    selector = RFECV(
        estimator=estimator,
        min_features_to_select=min_features,
        cv=cv,
        scoring='accuracy'
    )
    
    X_new = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.support_]
    X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)
    
    cv_results = {
        'n_features': selector.n_features_,
        'grid_scores': selector.grid_scores_,
        'ranking': selector.ranking_
    }
    
    return X_selected, cv_results 