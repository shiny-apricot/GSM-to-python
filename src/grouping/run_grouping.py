"""
Core grouping functionality for the GSM pipeline.

This module handles:
1. Loading and validating input data
2. Creating feature-group mappings
3. Organizing features into their respective groups

Key Functions:
- run_grouping: Main entry point that processes data and returns grouped features
"""
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass

import config
from .grouping_utils import create_group_feature_mapping
from .group_feature_mapping import GroupFeatureMappingData

def run_grouping(
    grouping_data: pd.DataFrame,
    **kwargs
) -> List[GroupFeatureMappingData]:
    """
    Run the grouping process to organize features into their respective groups.
    
    This function:
    1. Creates feature-to-group mappings
    2. Organizes features into their respective groups
    3. Returns grouped data and mappings
    
    Args:
        main_data: DataFrame containing features and labels
        grouping_data: DataFrame containing the grouping configuration
        **kwargs: Additional arguments for customizing the grouping process
            - custom_groups (List[str]): Specific groups to process
            - exclude_groups (List[str]): Groups to exclude from processing
    
    Returns:
        GroupingResult containing:
            - grouped_data: Dictionary mapping group names to their GroupData
            - feature_mappings: List of FeatureGroupMapping objects
    
    Raises:
        ValueError: If input data validation fails
    """
    # Create feature-to-group mapping
    feature_mappings = create_group_feature_mapping(grouping_data)
    
    # Get unique groups, applying any filters from kwargs
    # TODO: what is this?
    # groups = grouping_data[config.GROUP_COLUMN_NAME].unique()
    # if 'custom_groups' in kwargs:
    #     groups = [g for g in groups if g in kwargs['custom_groups']]
    # if 'exclude_groups' in kwargs:
    #     groups = [g for g in groups if g not in kwargs['exclude_groups']]
    
    # # Organize data by groups
    # grouped_data: Dict[str, GroupData] = {}
    # for group in groups:
    #     features, labels = get_group_data(main_data, grouping_data, group)
    #     grouped_data[group] = GroupData(
    #         features=features,
    #         labels=labels,
    #         group_name=group
    #     )
    
    # return GroupingResult(
    #     grouped_data=grouped_data,
    #     feature_mappings=feature_mappings
    # ) 

    return feature_mappings