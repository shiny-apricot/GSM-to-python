"""
Utility functions for the grouping module.

This module provides helper functions for:
- Creating feature-to-group mappings
- Extracting group-specific data
- Data transformation and validation
"""
from typing import List, Tuple
import pandas as pd

from config import GROUP_COLUMN_NAME, GENE_COLUMN_NAME, LABEL_COLUMN_NAME
from .group_feature_mapping import GroupFeatureMappingData

def create_group_feature_mapping(grouping_data: pd.DataFrame) -> List[GroupFeatureMappingData]:
    """
    Create feature-group mappings from input data.

    Args:
        grouping_data: DataFrame containing feature-group mapping information
            Must have columns: 'feature_name' and 'group_name'

    Returns:
        List of FeatureGroupMapping objects

    Raises:
        ValueError: If required columns are missing in the input DataFrame

    Example:
        >>> grouping_data = pd.DataFrame({
        ...     'gene': ['f1', 'f2'], 
        ...     'group': ['g1', 'g2']
        ... })
        >>> mappings = create_feature_group_mapping(grouping_data)
        >>> print(mappings[0])
        GroupFeatureMapping(group_name='g1', feature_list=['f1'])
    """
    # Check for required columns
    required_columns = [GENE_COLUMN_NAME, GROUP_COLUMN_NAME]
    for col in required_columns:
        if col not in grouping_data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create a dictionary to hold features for each group
    group_mapping = {}
    for _, row in grouping_data.iterrows():
        group_name = row[GROUP_COLUMN_NAME]
        feature_name = row[GENE_COLUMN_NAME]
        if group_name not in group_mapping:
            group_mapping[group_name] = []
        group_mapping[group_name].append(feature_name)

    return [
        GroupFeatureMappingData(group_name=group_name, feature_list=features)
        for group_name, features in group_mapping.items()
    ]

# def get_group_data(main_data: pd.DataFrame, 
#                   grouping_data: pd.DataFrame, 
#                   group: str) -> Tuple[pd.DataFrame, pd.Series]:
#     """
#     Extract features and labels for a specific group.

#     Args:
#         main_data: DataFrame containing all features and labels
#         grouping_data: DataFrame containing feature-group mappings
#         group: Name of the group to extract

#     Returns:
#         Tuple of (features DataFrame, labels Series) for the specified group
#     """
#     # Filter the grouping data for the specified group
#     group_features = grouping_data[grouping_data[GROUP_COLUMN_NAME] == group][GENE_COLUMN_NAME]
    
#     # Select the relevant features from the main data
#     features = main_data[group_features].copy()
#     labels = main_data[LABEL_COLUMN_NAME]  # Assuming 'label' is the column name for labels

#     return features, labels 