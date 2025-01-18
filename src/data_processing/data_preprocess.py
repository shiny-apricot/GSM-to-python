"""
Data Preprocessing Module for GSM Pipeline

This module handles the core data preprocessing tasks including:
- Loading input and group data files
- Converting labels to binary format
- Data normalization
- Train/test splitting

Key Functions:
- preprocess_data: Main preprocessing pipeline
- convert_labels_to_binary: Converts class labels to 0/1
- normalize_data: Applies normalization to features
- validate_input_data: Validates input data structure

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

from src.data_processing.data_loader import load_input_file, load_group_file
from config import LABEL_OF_NEGATIVE_CLASS, LABEL_OF_POSITIVE_CLASS
from normalization import normalize_data
from src.data_processing.train_test_splitter import train_test_split

def validate_input_data(data: pd.DataFrame) -> None:
    """
    Validates the structure and content of input data.
    
    Args:
        data: Input DataFrame to validate
        
    Raises:
        ValueError: If data format is invalid
    """
    if data.empty:
        raise ValueError("Input data is empty")
    
    required_columns = ['label']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def preprocess_data(
    project_folder: Union[str, Path],
    input_file_name: str,
    group_file_name: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing pipeline that handles all data preparation steps.
    
    Args:
        project_folder: Path to project directory
        input_file_name: Name of input data file
        group_file_name: Name of group data file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Preprocessed training data
        - Preprocessed test data
        - Group data
        
    Example:
        >>> train_data, test_data, group_data = preprocess_data(
        ...     project_folder="project/",
        ...     input_file_name="input.csv",
        ...     group_file_name="groups.csv"
        ... )
    """
    # Load data
    project_folder = Path(project_folder)
    input_data = load_input_file(project_folder, input_file_name)
    group_data = load_group_file(project_folder, group_file_name)
    
    # Validate
    validate_input_data(input_data)
    
    # Convert labels
    input_data = convert_labels_to_binary(
        input_data, 
        LABEL_OF_NEGATIVE_CLASS,
        LABEL_OF_POSITIVE_CLASS
    )
    
    # Normalize features
    normalized_data = normalize_data(input_data)
    
    # Split data
    train_data, test_data = train_test_split(
        normalized_data,
        test_size=test_size,
        random_state=random_state
    )
    
    return train_data, test_data, group_data

def convert_labels_to_binary(
    data: pd.DataFrame,
    negative_label: str = LABEL_OF_NEGATIVE_CLASS,
    positive_label: str = LABEL_OF_POSITIVE_CLASS
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
    if 'label' not in data.columns:
        raise ValueError("Data must contain 'label' column")
        
    label_map = {negative_label: 0, positive_label: 1}
    invalid_labels = set(data['label']) - set(label_map.keys())
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")
        
    data = data.copy()
    data['label'] = data['label'].map(label_map)
    return data