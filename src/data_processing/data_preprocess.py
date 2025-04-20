"""
🧬 Data Preprocessing Module for GSM Pipeline 🧬

Purpose:
    Handles core data preprocessing tasks for gene expression analysis in the GSM pipeline.

Primary Functions:
    🔍 preprocess_data: Main preprocessing pipeline coordinator
    🎯 convert_labels_to_binary: Binary label conversion (0/1)
    📊 normalize_data: Feature normalization
    ✅ validate_input_data: Input data validation
    📑 preprocess_grouping_data: Group data preprocessing

Input Data Requirements:
    - Must contain 'class' column for labels
    - Features should be numeric
    - No duplicate indices
    - Groups data must have GENE_COLUMN_NAME and GROUP_COLUMN_NAME

Example Usage:
    ```python
    from data_processing.data_preprocess import preprocess_data
    
    processed_train, processed_test = preprocess_data(
        input_data=input_df,
        label_column_name='class',
        label_of_negative_class='healthy',
        label_of_positive_class='disease',
        logger=logger,
        test_size=0.2
    )
    ```

Note: All functions use explicit parameter names for better code readability.
"""

from typing import Tuple, Union, Any
import pandas as pd

from data_processing.normalization import normalize_data
from data_processing.train_test_splitter import train_test_split
from data_processing.handle_missing_values import drop_missing_values, fill_missing_values
from config import (GENE_COLUMN_NAME, 
                    GROUP_COLUMN_NAME, 
                    MIN_CLASS_BALANCE_RATIO, 
                    SAMPLING_METHOD)


def validate_input_data(data: pd.DataFrame, label_column_name: str) -> None:
    """
    Validates input data structure and content.
    
    Parameters:
        data (pd.DataFrame): Input data to validate
        label_column_name (str): Name of the label column
    
    Raises:
        ValueError: With detailed message if validation fails
            - Empty DataFrame
            - Missing required columns
            - Invalid data types
    """
    if data.empty:
        raise ValueError("Input data is empty")
    
    required_columns = ['class']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def preprocess_data(
    input_data: pd.DataFrame,
    label_column_name: str,
    label_of_negative_class: str,
    label_of_positive_class: str,
    logger: Any,
    test_size: float = 0.2,
    normalization_method: str = 'zscore',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Executes the complete data preprocessing pipeline.
    
    Parameters:
        input_data (pd.DataFrame): Raw input data
        label_column_name (str): Name of the label column
        label_of_negative_class (str): Label representing negative class
        label_of_positive_class (str): Label representing positive class
        logger (Any): Logger instance for tracking progress
        test_size (float, optional): Proportion of test set. Defaults to 0.2
        normalization_method (str, optional): Method for normalization. Defaults to 'zscore'
        random_state (int, optional): Random seed. Defaults to 42
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Processed (train_data, test_data)
    
    Raises:
        ValueError: If input validation fails
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
    sampled_data = determine_class_balance(normalized_data, 
                                           logger=logger,
                                           min_class_balance_ratio=0.5,
                                           sampling_method='undersampling')
    logger.info(f"Sampled data size: {sampled_data.shape}")
    
    # Split data
    # train_data, test_data = train_test_split(
    #     normalized_data,
    #     test_size=test_size,
    #     random_state=random_state
    # )
    
    # Give a snapshot of the data
    logger.info(f"Data shape: {sampled_data.shape}")
    logger.info(f"Data columns: {sampled_data.columns.tolist()}")
    logger.info(f"Data head:\n{sampled_data.head()}")
    
    return sampled_data

def convert_labels_to_binary(
    data: pd.DataFrame,
    label_column_name: str,
    negative_label: str,
    positive_label: str
) -> pd.DataFrame:
    """
    Converts categorical labels to binary format.
    
    Parameters:
        data (pd.DataFrame): Input data with labels
        label_column_name (str): Name of label column
        negative_label (str): Label to convert to 0
        positive_label (str): Label to convert to 1
    
    Returns:
        pd.DataFrame: Data with binary labels
    
    Raises:
        ValueError: If label column is missing or invalid labels found
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

   
def determine_class_balance(data: pd.DataFrame,
                            logger: Any,
                            min_class_balance_ratio: float = 0.5,
                            sampling_method: str = 'undersampling') -> pd.DataFrame:
    """
    Determines class balance in the dataset and applies sampling if necessary.

    Parameters:
        data (pd.DataFrame): Input data with labels
        label1 (str): Name of the first label column
        label2 (str): Name of the second label column
        logger (Any): Logger instance
        min_class_balance_ratio (float): Minimum acceptable ratio between minority and majority classes
            (0.5 means classes can be at most 1:2)
        sampling_method (str): Method for balancing classes ('undersampling', 'oversampling')
    Returns:
        pd.DataFrame: Data with balanced classes
    """
    negative_class = "0"
    positive_class = "1"
    class_label = 'class'

    # Count occurrences of each class
    class_counts = data[class_label].value_counts()
    logger.info(f"Class counts: {class_counts}")
    
    # Check if classes are balanced
    if class_counts.min() / class_counts.max() < 0.5:
        # Apply sampling method to balance classes
        if sampling_method == 'undersampling':
            # Find the minority and majority classes
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            
            # Get all samples from the minority class
            minority_samples = data[data[class_label] == minority_class]
            
            # Randomly sample from the majority class to match minority class count
            majority_samples = data[data[class_label] == majority_class].sample(
                n=class_counts.min(),
            )
            
            # Combine minority and sampled majority classes
            data = pd.concat([minority_samples, majority_samples])
        elif sampling_method == 'oversampling':
            # Find the minority class (the class with the fewest samples)
            minority_class = class_counts.idxmin()
            # Create copies of the minority class data and append them to balance the classes
            # This duplicates minority class samples until they roughly match the majority class
            data = pd.concat([data, data[data[class_label] == minority_class].copy()] * (class_counts.max() // class_counts.min()))
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")
    else:
        logger.info("Classes are balanced, no sampling applied")
    logger.info(f"Balanced class counts: {data[class_label].value_counts()}")
    # Check if the class balance ratio is acceptable
    ratio = data[class_label].value_counts().min() / data[class_label].value_counts().max()
    if ratio < min_class_balance_ratio:
        logger.warning(f"Class balance ratio {ratio} is below the minimum threshold {min_class_balance_ratio}")
    else:
        logger.info(f"Class balance ratio {ratio} is acceptable")
    return data

def preprocess_grouping_data(grouping_data: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """
    Preprocesses gene grouping data by validating its structure.
    
    Parameters:
        grouping_data (pd.DataFrame): Raw grouping data
        logger (Any): Logger instance
    
    Returns:
        pd.DataFrame: Processed grouping data
    
    Raises:
        ValueError: If data is empty or missing required columns
    """
    try:        
        # Validate the structure
        if grouping_data.empty:
            raise ValueError("Grouping data is empty")
        
        required_columns = [GENE_COLUMN_NAME, GROUP_COLUMN_NAME]
        missing_cols = [col for col in required_columns if col not in grouping_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Give a snapshot of the data
        logger.info(f"Grouping data shape: {grouping_data.shape}")
        logger.info(f"Grouping data columns: {grouping_data.columns.tolist()}")
        logger.info(f"Grouping data head:\n{grouping_data.head()}")
        
        return grouping_data

    except Exception as e:
        logger.error(f"Error processing grouping data: {e}")
        raise