"""
📊 Group-Feature Mapping Module
================================

This module handles the mapping between features (genes) and their corresponding groups.

Key Components:
--------------
- GroupFeatureMappingData: Dataclass for storing group-feature relationships

Usage Example:
-------------
>>> mapping = GroupFeatureMappingData(
...     group_name="pathway_001",
...     feature_list=["gene1", "gene2", "gene3"]
... )
>>> print(mapping.group_name)
'pathway_001'
"""

from dataclasses import dataclass
from typing import List

@dataclass
class GroupFeatureMappingData:
    """
    Maps features (e.g., genes) to their corresponding groups (e.g., pathways).
    
    Attributes:
    -----------
    group_name : str
        Unique identifier for the group (e.g., pathway name, GO term)
    feature_list : List[str]
        List of feature identifiers belonging to this group
        
    Example:
    --------
    >>> group = GroupFeatureMappingData("DNA_REPAIR", ["BRCA1", "BRCA2", "RAD51"])
    """
    group_name: str
    feature_list: List[str]

    def __post_init__(self):
        """Validate the data after initialization"""
        if not isinstance(self.group_name, str):
            raise TypeError("group_name must be a string")
        if not isinstance(self.feature_list, list):
            raise TypeError("feature_list must be a list")
        if not all(isinstance(f, str) for f in self.feature_list):
            raise TypeError("All features must be strings")
        if not self.feature_list:
            raise ValueError("feature_list cannot be empty")
