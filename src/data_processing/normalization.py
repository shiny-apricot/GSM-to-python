"""
Normalization Module
-------------------
This module handles data normalization operations in the GSM pipeline.

Key Functions:
- normalize_data: Main function for normalizing input data
- _minmax_normalize: Min-max normalization implementation
- _zscore_normalize: Z-score normalization implementation
- _robust_normalize: Robust scaling implementation

Usage Example:
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> normalized = normalize_data(data, method='zscore')
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def _validate_input(data: pd.DataFrame) -> None:
    """Validate input data for normalization."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    if data.isnull().any().any():
        raise ValueError("Input DataFrame contains missing values")
    if not np.issubdtype(data.values.dtype, np.number):
        raise ValueError("All columns must contain numeric values")

def _minmax_normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Apply min-max normalization."""
    scaler = MinMaxScaler()
    return pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

def _zscore_normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Apply z-score normalization."""
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

def _robust_normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Apply robust scaling using median and IQR."""
    scaler = RobustScaler()
    return pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

def normalize_data(
    data: pd.DataFrame,
    label_column_name: str,
    logger,
    method: str = 'zscore',
    subset_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Normalize the input data using the specified method.

    Args:
        data (pd.DataFrame): Input data to normalize
        label_column_name (str): Name of the label column to exclude from normalization
        method (str): Normalization method ('minmax', 'zscore', or 'robust')
        subset_cols (list, optional): Specific columns to normalize. If None, normalize all columns except the label

    Returns:
        pd.DataFrame: Normalized data
    """
    logger.info(f"Starting data normalization using {method} method")
    
    # Input validation
    _validate_input(data)
    
    # Create a copy to avoid modifying the original data
    data_to_normalize = data.copy()
    
    # Select columns to normalize
    if subset_cols is None:
        cols_to_normalize = data.columns.difference([label_column_name])  # Exclude label column
    else:
        cols_to_normalize = [col for col in subset_cols if col != label_column_name]  # Exclude label column if in subset
    
    # Validate selected columns
    if not all(col in data.columns for col in cols_to_normalize):
        raise ValueError("One or more specified columns not found in DataFrame")
    
    try:
        # Apply normalization based on method
        normalization_methods = {
            'minmax': _minmax_normalize,
            'zscore': _zscore_normalize,
            'robust': _robust_normalize
        }
        
        if method not in normalization_methods:
            raise ValueError(
                f"Invalid normalization method. Choose from: {', '.join(normalization_methods.keys())}"
            )
        
        # Normalize selected columns
        data_to_normalize[cols_to_normalize] = normalization_methods[method](
            data_to_normalize[cols_to_normalize]
        )
        
        logger.info("Data normalization completed successfully")
        return data_to_normalize
        
    except Exception as e:
        logger.error(f"Error during normalization: {str(e)}")
        raise