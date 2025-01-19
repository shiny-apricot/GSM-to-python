"""
Data structures used in the grouping module.

This module contains dataclass definitions that represent:
- Feature to group mappings
- Grouped data structures
- Results from the grouping process
"""

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

@dataclass
class GroupFeatureMapping:
    """Maps features to their corresponding groups"""
    group_name: str
    feature_list: List[str]

@dataclass
class GroupData:
    """Holds feature data and labels for a group"""
    features: pd.DataFrame
    labels: pd.Series
    group_name: str

@dataclass
class GroupingResult:
    """Contains all results from the grouping process"""
    grouped_data: Dict[str, GroupData]
    feature_mappings: List[GroupFeatureMapping] 