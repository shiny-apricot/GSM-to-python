"""
File's primary purpose and role in the pipeline:
This module provides functions to handle missing values in datasets, ensuring data integrity for subsequent analysis steps.

Key functions:
- drop_missing_values: Removes rows with missing values from the dataset.
- fill_missing_values: Fills missing values using specified strategies (mean, median, mode).

Usage examples:
- cleaned_data = drop_missing_values(data)
- filled_data = fill_missing_values(data, strategy='mean')

Important notes:
- Ensure that the chosen strategy for filling missing values aligns with the data characteristics.
"""

import pandas as pd

def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with any missing values from the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with rows containing missing values removed.
    """
    return data.dropna()

def fill_missing_values(data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Fills missing values in the DataFrame using the specified strategy.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    strategy (str): The strategy to use for filling missing values ('mean', 'median', 'mode').

    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'mode'.")
