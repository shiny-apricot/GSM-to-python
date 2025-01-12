# grouping/group_validator.py

import pandas as pd
from .input_loader import load_group_file, get_unique_group_names
from .group_mapper import get_features_of_each_group

class GroupValidator:
    def __init__(self, project_folder, group_file_name):
        self.project_folder = project_folder
        self.group_file_name = group_file_name
        self.group_data = load_group_file(project_folder, group_file_name)
        self.unique_groups = get_unique_group_names(self.group_data)

    def validate_group_names(self):
        """
        Validate the group names.

        Returns:
            list: A list of invalid group names.
        """
        invalid_group_names = []
        for group_name in self.unique_groups:
            if not isinstance(group_name, str):
                invalid_group_names.append(group_name)
        return invalid_group_names

    def validate_feature_names(self):
        """
        Validate the feature names.

        Returns:
            list: A list of invalid feature names.
        """
        invalid_feature_names = []
        group_features = get_features_of_each_group(self.group_data)
        for group_name, features in group_features.items():
            for feature in features:
                if not isinstance(feature, str):
                    invalid_feature_names.append(feature)
        return invalid_feature_names

    def validate_group_feature_mapping(self):
        """
        Validate the group-feature mapping.

        Returns:
            list: A list of invalid group-feature mappings.
        """
        invalid_mappings = []
        group_features = get_features_of_each_group(self.group_data)
        for group_name, features in group_features.items():
            if not features:
                invalid_mappings.append((group_name, features))
        return invalid_mappings