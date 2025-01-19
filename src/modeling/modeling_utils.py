"""
Utility functions for data preprocessing and model preparation in the GSM pipeline.

Key Functions:
- preprocess_features: Standardizes numeric features and encodes categorical variables
- filter_best_groups: Filters gene groups based on scoring thresholds
- validate_data: Validates input data structure and types
- handle_missing_values: Implements strategies for handling missing data
"""

from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validates input DataFrame structure and content.
    
    Args:
        data: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if data.empty:
        return False, "DataFrame is empty"
        
    if data.isnull().all().any():
        return False, "DataFrame contains columns with all null values"
        
    return True, ""

def handle_missing_values(
    data: pd.DataFrame,
    strategy: str = 'mean',
    fill_value: Optional[Union[str, float]] = None
) -> pd.DataFrame:
    """
    Handles missing values in the dataset.
    
    Args:
        data: Input DataFrame
        strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'constant')
        fill_value: Value to use if strategy is 'constant'
        
    Returns:
        DataFrame with handled missing values
    """
    numeric_data = data.select_dtypes(include=[np.number])
    categorical_data = data.select_dtypes(include=['object', 'category'])
    
    # Handle numeric columns
    if not numeric_data.empty:
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        data[numeric_data.columns] = imputer.fit_transform(numeric_data)
    
    # Handle categorical columns
    if not categorical_data.empty:
        data[categorical_data.columns] = categorical_data.fillna('missing')
    
    return data

def preprocess_features(
    data: pd.DataFrame,
    logger,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    scaler_type: str = 'standard',
) -> pd.DataFrame:
    """
    Preprocess features by scaling numeric columns and encoding categorical ones.
    
    Args:
        data: Input DataFrame
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
        scaler_type: Type of scaler to use ('standard' or 'robust')
        handle_nulls: Whether to handle missing values before preprocessing
        
    Returns:
        Preprocessed DataFrame
        
    Raises:
        ValueError: If input validation fails or invalid parameters are provided
    """
    # Validate input data
    is_valid, error_message = validate_data(data)
    if not is_valid:
        raise ValueError(error_message)
    
    # Create copy to avoid modifying original data
    processed_data = data.copy()
    
    # Identify column types if not provided
    if numeric_columns is None:
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
    if categorical_columns is None:
        categorical_columns = processed_data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
    
    # Scale numeric features
    if numeric_columns:
        scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        try:
            processed_data[numeric_columns] = scaler.fit_transform(
                processed_data[numeric_columns]
            )
        except Exception as e:
            logger.error(f"Error scaling numeric features: {str(e)}")
            raise
    
    # Encode categorical features
    for col in categorical_columns:
        try:
            processed_data[col] = pd.Categorical(processed_data[col]).codes
        except Exception as e:
            logger.error(f"Error encoding categorical column {col}: {str(e)}")
            raise
        
    return processed_data

def filter_best_groups(
    group_scores: pd.DataFrame,
    threshold: float = 0.75,
    min_groups: int = 1,
    max_groups: Optional[int] = None
) -> List[str]:
    """
    Filter groups based on their scores with additional controls.
    
    Args:
        group_scores: DataFrame with group scores
        threshold: Score threshold for filtering (0.0 to 1.0)
        min_groups: Minimum number of groups to return
        max_groups: Maximum number of groups to return
        
    Returns:
        List of group names that meet the criteria
        
    Raises:
        ValueError: If parameters are invalid or no groups meet criteria
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    if not isinstance(group_scores, pd.DataFrame):
        raise ValueError("group_scores must be a pandas DataFrame")
    
    if 'score' not in group_scores.columns:
        raise ValueError("group_scores must contain a 'score' column")
    
    # Get groups meeting threshold
    filtered_groups = group_scores[group_scores['score'] >= threshold].index.tolist()
    
    # Adjust threshold if minimum groups not met
    if len(filtered_groups) < min_groups:
        logger.warning(
            f"Not enough groups meet threshold. Adjusting threshold to get {min_groups} groups"
        )
        filtered_groups = group_scores.nlargest(min_groups, 'score').index.tolist()
    
    # Limit maximum groups if specified
    if max_groups is not None and len(filtered_groups) > max_groups:
        filtered_groups = group_scores.nlargest(max_groups, 'score').index.tolist()
    
    return filtered_groups 