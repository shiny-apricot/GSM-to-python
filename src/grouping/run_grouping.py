"""
🧬 Core Grouping Module for GSM Pipeline 🧬

This module organizes features into logical groups based on biological relationships
or other criteria specified in the grouping configuration.

Key Functions:
-------------
- run_grouping: Main entry point for grouping process
- validate_grouping_data: Validates input data format and content

Example Usage:
-------------
>>> grouping_data = pd.read_csv('group_config.csv')
>>> grouped_features = run_grouping(grouping_data)
>>> print(f"Created {len(grouped_features)} feature groups")
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from .grouping_utils import create_group_feature_mapping
from .group_feature_mapping import GroupFeatureMappingData


# TODO: remove config from this module
@dataclass
class GroupingParameters:
    """Configuration for grouping process."""
    custom_groups: Optional[List[str]] = None
    exclude_groups: Optional[List[str]] = None
    min_features_per_group: int = 3


def run_grouping(
    grouping_data: pd.DataFrame,
    group_feature_mappings: List[GroupFeatureMappingData],
    logger,
    params: Optional[GroupingParameters] = None
) -> List[GroupFeatureMappingData]:
    """
    Run the grouping process to organize features into their respective groups.
    
    Args:
        grouping_data: DataFrame with columns ['feature_id', 'group_name']
        config: Optional GroupingConfig for customizing the process
    
    Returns:
        List[GroupFeatureMappingData]: Organized feature groups
    
    Raises:
        ValueError: If input validation fails
    """
    logger.info("##### Starting grouping process #####")
    
    try:
        # Validate input data
        validate_grouping_data(grouping_data, logger)
        
        # Create feature-to-group mapping
        logger.info("Creating feature-group mappings...")
        
        # Filter groups based on config.custom_groups if specified
        # Only keep groups that are in the custom_groups list
        if params and params.custom_groups:
            selected_feature_groups = [
            group_mapping for group_mapping in group_feature_mappings 
            if group_mapping.group_name in params.custom_groups
            ]
            group_feature_mappings = selected_feature_groups
        
        # Remove any groups specified in config.exclude_groups
        if params and params.exclude_groups:
            filtered_feature_groups = [
            group_mapping for group_mapping in group_feature_mappings 
            if group_mapping.group_name not in params.exclude_groups
            ]
            group_feature_mappings = filtered_feature_groups
        
        # Validate results
        if not group_feature_mappings:
            raise ValueError("❌ No valid feature groups created")
        
        logger.info(f"✅ Successfully created {len(group_feature_mappings)} feature groups")
        return group_feature_mappings
        
    except Exception as e:
        logger.error(f"❌ Error in grouping process: {str(e)}")
        raise


def validate_grouping_data(grouping_data: pd.DataFrame,
                           logger) -> None:
    """
    Validate the grouping data format and content.
    
    Args:
        grouping_data: DataFrame containing grouping configuration

    Raises:
        ValueError: If validation fails
    """
    # TODO: check required columns in grouping data
    
    # Check for empty data 
    if grouping_data.empty:
        raise ValueError("❌ Grouping data is empty")
    
    logger.info("✅ Grouping data validation successful")