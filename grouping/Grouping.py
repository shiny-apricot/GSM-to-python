import pandas as pd


class Grouping:
    """
    A class to handle grouping of features and labels.
    """

    def __init__(self, 
                 group_data: pd.DataFrame, 
                 feature_data: pd.DataFrame, 
                 label_data: pd.Series):
        """
        Initialize the Grouping class.

        Args:
            group_data (pd.DataFrame): The group data.
            feature_data (pd.DataFrame): The feature data.
            label_data (pd.Series): The label data.
        """
        self.group_data = group_data
        self.feature_data = feature_data
        self.label_data = label_data
        
    def run_grouping(self, group_data, feature_data, label_data):
        """
        Run the grouping process.

        Returns:
            tuple: A tuple containing the group features, group labels, and feature groups.
        """
        group_features = self.group_data["feature_name"].tolist()
        group_labels = self.group_data["label"].tolist()
        feature_group_mapping = self.feature_data["feature_name"].tolist()
        return group_features, group_labels, feature_group_mapping

    def get_group_features(self, group_name):
        """
        Get the features for a given group.

        Args:
            group_name (str): The name of the group.

        Returns:
            list: A list of feature names for the given group.
        """
        group_features = self.group_data[self.group_data["group_name"] == group_name]["feature_name"].tolist()
        return group_features

    def get_group_labels(self, group_name):
        """
        Get the labels for a given group.

        Args:
            group_name (str): The name of the group.

        Returns:
            list: A list of labels for the given group.
        """
        group_labels = self.group_data[self.group_data["group_name"] == group_name]["label"].tolist()
        return group_labels

    def get_feature_groups(self, feature_name):
        """
        Get the groups for a given feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            list: A list of group names for the given feature.
        """
        feature_groups = self.group_data[self.group_data["feature_name"] == feature_name]["group_name"].tolist()
        return feature_groups

    def get_feature_labels(self, feature_name):
        """
        Get the labels for a given feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            list: A list of labels for the given feature.
        """
        feature_labels = self.feature_data[self.feature_data["feature_name"] == feature_name]["label"].tolist()
        return feature_labels
