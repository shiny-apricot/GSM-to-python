"""
T-test based feature selection for gene expression data.

This module implements t-test based filtering to identify differentially expressed genes
between two conditions (e.g., disease vs. control). It provides functions for:
- Performing t-tests on gene expression data
- Filtering genes based on p-value thresholds
- Handling multiple testing corrections

Key Functions:
    - perform_ttest: Executes t-test analysis on gene expression data
    - filter_by_pvalue: Filters genes based on calculated p-values
    - adjust_pvalues: Applies multiple testing corrections
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TTestResults:
    """Container for t-test results."""
    statistics: np.ndarray
    pvalues: np.ndarray
    adjusted_pvalues: np.ndarray
    selected_features: np.ndarray

def perform_ttest(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    equal_var: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform t-test for each feature between two groups.

    Args:
        X: Gene expression data (samples × features)
        y: Binary labels indicating groups (0 or 1)
        equal_var: Whether to assume equal variances (default: False)

    Returns:
        Tuple of (t-statistics, p-values)

    Example:
        >>> X = np.random.randn(100, 1000)  # 100 samples, 1000 genes
        >>> y = np.random.binomial(1, 0.5, 100)  # Binary labels
        >>> t_stats, p_vals = perform_ttest(X, y)
    """
    try:
        # Convert inputs to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)

        # Validate inputs
        if len(np.unique(y)) != 2:
            raise ValueError("y must contain exactly two unique classes")

        # Split data into two groups
        group1_data = X[y == 0]
        group2_data = X[y == 1]

        # Perform t-test for each feature
        t_stats, p_vals = stats.ttest_ind(
            group1_data, 
            group2_data, 
            equal_var=equal_var,
            axis=0
        )

        return t_stats, p_vals

    except Exception as e:
        logger.error(f"Error in perform_ttest: {str(e)}")
        raise

def adjust_pvalues(
    pvalues: np.ndarray,
    method: str = 'fdr_bh'
) -> np.ndarray:
    """
    Apply multiple testing correction to p-values.

    Args:
        pvalues: Array of p-values
        method: Correction method ('fdr_bh', 'bonferroni', etc.)

    Returns:
        Array of adjusted p-values
    """
    try:
        return stats.multipletests(pvalues, method=method)[1]
    except Exception as e:
        logger.error(f"Error in adjust_pvalues: {str(e)}")
        raise

def filter_by_pvalue(
    X: Union[np.ndarray, pd.DataFrame],
    pvalues: np.ndarray,
    threshold: float = 0.05,
    feature_names: Union[np.ndarray, list] = None
) -> TTestResults:
    """
    Filter features based on p-values.

    Args:
        X: Original feature matrix
        pvalues: P-values for each feature
        threshold: P-value threshold for filtering
        feature_names: Optional list of feature names

    Returns:
        TTestResults object containing filtered features and statistics
    """
    try:
        # Adjust p-values for multiple testing
        adjusted_pvals = adjust_pvalues(pvalues)
        
        # Select features meeting threshold
        selected_mask = adjusted_pvals < threshold
        
        if feature_names is not None:
            selected_features = np.array(feature_names)[selected_mask]
        else:
            selected_features = np.arange(len(pvalues))[selected_mask]

        logger.info(f"Selected {sum(selected_mask)} features using p-value threshold {threshold}")
        
        return TTestResults(
            statistics=pvalues,
            pvalues=pvalues,
            adjusted_pvalues=adjusted_pvals,
            selected_features=selected_features
        )

    except Exception as e:
        logger.error(f"Error in filter_by_pvalue: {str(e)}")
        raise

def select_features(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.05,
    feature_names: Union[np.ndarray, list] = None,
    equal_var: bool = False
) -> TTestResults:
    """
    Complete pipeline for t-test based feature selection.

    Args:
        X: Gene expression data
        y: Binary class labels
        threshold: P-value threshold for filtering
        feature_names: Optional list of feature names
        equal_var: Whether to assume equal variances

    Returns:
        TTestResults object with selected features and statistics
    """
    try:
        # Perform t-test
        t_stats, p_vals = perform_ttest(X, y, equal_var=equal_var)
        
        # Filter features
        results = filter_by_pvalue(
            X, p_vals, 
            threshold=threshold,
            feature_names=feature_names
        )
        
        return results

    except Exception as e:
        logger.error(f"Error in select_features: {str(e)}")
        raise 