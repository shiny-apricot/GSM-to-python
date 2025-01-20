
from dataclasses import dataclass
from typing import List

@dataclass
class GroupFeatureMappingData:
    """Maps features to their corresponding groups"""
    group_name: str
    feature_list: List[str]
