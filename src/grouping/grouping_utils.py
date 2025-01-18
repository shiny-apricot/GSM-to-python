"""
Utility functions for the grouping module.

This module provides helper functions for:
- Creating feature-to-group mappings
- Extracting group-specific data
- Data transformation and validation
"""
from typing import List, Tuple
import pandas as pd
from .data_structures import FeatureGroupMapping, GroupData

def create_feature_group_mapping(grouping_data: pd.DataFrame) -> List[FeatureGroupMapping]:
    """
    Create feature-group mappings from input data.

    Args:
        grouping_data: DataFrame containing feature-group mapping information
            Must have columns: 'feature_name' and 'group_name'

    Returns:
        List of FeatureGroupMapping objects

    Example:
        >>> grouping_data = pd.DataFrame({
        ...     'feature_name': ['f1', 'f2'], 
        ...     'group_name': ['g1', 'g2']
        ... })
        >>> mappings = create_feature_group_mapping(grouping_data)
        >>> print(mappings[0])
        FeatureGroupMapping(feature_name='f1', group_name='g1')
    """
    return [
        FeatureGroupMapping(feature_name=row['feature_name'], group_name=row['group_name'])
        for _, row in grouping_data.iterrows()
    ]

def get_group_data(main_data: pd.DataFrame, 
                  grouping_data: pd.DataFrame, 
                  group: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and labels for a specific group.

    Args:
        main_data: DataFrame containing all features and labels
        grouping_data: DataFrame containing feature-group mappings
        group: Name of the group to extract

    Returns:
        Tuple of (features DataFrame, labels Series) for the specified group
    """
    # ... existing implementation ... 