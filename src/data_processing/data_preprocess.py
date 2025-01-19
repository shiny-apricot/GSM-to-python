"""
Data Preprocessing Module for GSM Pipeline

This module handles the core data preprocessing tasks including:
- Loading input and group data files
- Converting labels to binary format
- Data normalization
- Train/test splitting
- Preprocessing grouping data from a .txt file

Key Functions:
- preprocess_data: Main preprocessing pipeline
- convert_labels_to_binary: Converts class labels to 0/1
- normalize_data: Applies normalization to features
- validate_input_data: Validates input data structure
- preprocess_grouping_data: Loads and preprocesses grouping data from a .txt file

Usage Example:
    preprocessed_data = preprocess_data(
        project_folder="path/to/project",
        input_file="input.csv",
        group_file="groups.csv"
    )
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd
import numpy as np

from data_processing.data_loader import load_input_file, load_group_file
from data_processing.normalization import normalize_data
from data_processing.train_test_splitter import train_test_split
from data_processing.handle_missing_values import drop_missing_values, fill_missing_values
from config import GENE_COLUMN_NAME, GROUP_COLUMN_NAME


def validate_input_data(data: pd.DataFrame, label_column_name: str) -> None:
    """
    Validates the structure and content of input data.
    
    Args:
        data: Input DataFrame to validate
        
    Raises:
        ValueError: If data format is invalid
    """
    if data.empty:
        raise ValueError("Input data is empty")
    
    required_columns = ['class']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def preprocess_data(
    input_data: pd.DataFrame,
    label_column_name:str,
    label_of_negative_class,
    label_of_positive_class,
    logger,
    test_size: float = 0.2,
    normalization_method='zscore',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing pipeline that handles all data preparation steps.
    
    Args:
        input_data: DataFrame containing input data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Preprocessed training data
        - Preprocessed test data
        
    Example:
        >>> train_data, test_data = preprocess_data(
        ...     input_data=df,
        ...     group_data=group_df
        ... )
    """
    # Validate
    validate_input_data(input_data, label_column_name)
    
    # handle missing values
    input_data = drop_missing_values(input_data)
    
    # Convert labels
    input_data = convert_labels_to_binary(
        input_data, 
        label_column_name,
        label_of_negative_class,
        label_of_positive_class
    )
    
    # Normalize features
    normalized_data = normalize_data(input_data,
                                     label_column_name=label_column_name,
                                     logger=logger,
                                     method=normalization_method)
    
    # Split data
    train_data, test_data = train_test_split(
        normalized_data,
        test_size=test_size,
        random_state=random_state
    )
    
    return train_data, test_data

def convert_labels_to_binary(
    data: pd.DataFrame,
    label_column_name,
    negative_label: str,
    positive_label: str
) -> pd.DataFrame:
    """
    Converts categorical labels to binary (0/1) format.
    
    Args:
        data: Input DataFrame with 'label' column
        negative_label: Label to convert to 0
        positive_label: Label to convert to 1
        
    Returns:
        DataFrame with converted binary labels
        
    Example:
        >>> df = pd.DataFrame({'label': ['healthy', 'sick']})
        >>> convert_labels_to_binary(df, 'healthy', 'sick')
           label
        0     0
        1     1
    """
    if label_column_name not in data.columns:
        raise ValueError("Data must contain 'label' column")
        
    label_map = {negative_label: 0, positive_label: 1}
    invalid_labels = set(data[label_column_name]) - set(label_map.keys())
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")
        
    data = data.copy()
    data[label_column_name] = data[label_column_name].map(label_map)
    return data

   
def sample_by_ratio(data: pd.DataFrame, ratio: float = 0.5) -> pd.DataFrame:
    """
    Samples a subset of the data based on a specified ratio.

    Args:
        data: Input DataFrame to sample from
        ratio: Sampling ratio (default: 0.5)

    Returns:
        Sampled DataFrame

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        >>> sampled_df = sample_by_ratio(df, 0.5)
    """
    if not 0 < ratio < 1:
        raise ValueError("Sampling ratio must be between 0 and 1")
    sample_size = int(len(data) * ratio)
    return data.sample(sample_size)

def preprocess_grouping_data(grouping_data:pd.DataFrame, logger) -> pd.DataFrame:
    """
    Loads and preprocesses grouping data from a .txt file.

    Args:
        file_path: Path to the grouping data file

    Returns:
        DataFrame containing the processed grouping data

    Example:
        >>> grouping_data = preprocess_grouping_data("data/grouping_file/cancer-DisGeNET.txt")
    """
    try:        
        # Validate the structure
        if grouping_data.empty:
            raise ValueError("Grouping data is empty")
        
        required_columns = [GENE_COLUMN_NAME, GROUP_COLUMN_NAME]
        missing_cols = [col for col in required_columns if col not in grouping_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return grouping_data

    except Exception as e:
        logger.error(f"Error processing grouping data: {e}")
        raise