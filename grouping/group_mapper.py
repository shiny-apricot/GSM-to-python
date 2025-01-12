# group_mapper.py

import pandas as pd

class GroupMapper:
    """
    A class for mapping groups to features and vice versa.
    E.g. group "Glioblastoma" is mapped to features "A1BG", "NAT2, "SERPINA3".... etc.
    """
    def __init__(self, group_data):
        """
        Initialize the GroupMapper class.

        Args:
            group_data (pd.DataFrame): The group data.
        """
        self.group_data = group_data

    def map_groups_to_features(self):
        """
        Map groups to features.

        Returns:
            dict: A dictionary where the keys are group names and the values are lists of feature names.
        """
        group_features = {}
        for group_name in self.group_data["group_name"].unique():
            features = self.group_data[self.group_data["group_name"] == group_name]["feature_name"].tolist()
            group_features[group_name] = features
        return group_features

    def map_features_to_groups(self):
        """
        Map features to groups.

        Returns:
            dict: A dictionary where the keys are feature names and the values are lists of group names.
        """
        feature_groups = {}
        for feature_name in self.group_data["feature_name"].unique():
            groups = self.group_data[self.group_data["feature_name"] == feature_name]["group_name"].tolist()
            feature_groups[feature_name] = groups
        return feature_groups

    def get_group_features(self, group_name):
        """
        Get the features for a given group.

        Args:
            group_name (str): The name of the group.

        Returns:
            list: A list of feature names for the given group.
        """
        return self.map_groups_to_features().get(group_name, [])

    def get_feature_groups(self, feature_name):
        """
        Get the groups for a given feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            list: A list of group names for the given feature.
        """
        return self.map_features_to_groups().get(feature_name, [])