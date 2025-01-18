"""
Core grouping functionality for the GSM pipeline.

This module handles:
1. Loading and validating input data
2. Creating feature-group mappings
3. Organizing features into their respective groups

Key Functions:
- run_grouping: Main entry point that processes data and returns grouped features
"""
from typing import Dict, List
import pandas as pd
from .data_structures import GroupData, GroupingResult
from .grouping_utils import create_feature_group_mapping, get_group_data
from .input_loader import prepare_data

def run_grouping(
    main_data_path: str,
    grouping_data_path: str,
    **kwargs
) -> GroupingResult:
    """
    Run the grouping process to organize features into their respective groups.
    
    This function:
    1. Loads and validates input data
    2. Creates feature-to-group mappings
    3. Organizes features into their respective groups
    4. Returns grouped data and mappings
    
    Args:
        main_data_path: Path to the main data file containing features and labels
        grouping_data_path: Path to the grouping configuration file
        **kwargs: Additional arguments for customizing the grouping process
            - custom_groups (List[str]): Specific groups to process
            - exclude_groups (List[str]): Groups to exclude from processing
    
    Returns:
        GroupingResult containing:
            - grouped_data: Dictionary mapping group names to their GroupData
            - feature_mappings: List of FeatureGroupMapping objects
    
    Raises:
        ValueError: If input data validation fails
        FileNotFoundError: If input files cannot be found
    """
    # Load and validate input data
    main_data, grouping_data = prepare_data(main_data_path, grouping_data_path)
    
    # Create feature-to-group mapping
    feature_mappings = create_feature_group_mapping(grouping_data)
    
    # Get unique groups, applying any filters from kwargs
    groups = grouping_data['group_name'].unique()
    if 'custom_groups' in kwargs:
        groups = [g for g in groups if g in kwargs['custom_groups']]
    if 'exclude_groups' in kwargs:
        groups = [g for g in groups if g not in kwargs['exclude_groups']]
    
    # Organize data by groups
    grouped_data: Dict[str, GroupData] = {}
    for group in groups:
        features, labels = get_group_data(main_data, grouping_data, group)
        grouped_data[group] = GroupData(
            features=features,
            labels=labels,
            group_name=group
        )
    
    return GroupingResult(
        grouped_data=grouped_data,
        feature_mappings=feature_mappings
    ) 