# group_scorer.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class GroupScorer:
    def __init__(self, group_data, predictions):
        """
        Initialize the GroupScorer class.

        Args:
            group_data (pd.DataFrame): The group data.
            predictions (pd.Series): The predicted labels.
        """
        self.group_data = group_data
        self.predictions = predictions

    def calculate_group_scores(self):
        """
        Calculate the scores for each group.

        Returns:
            dict: A dictionary where the keys are group names and the values are dictionaries containing the accuracy, precision, recall, and F1 score.
        """
        group_scores = {}
        for group_name in self.group_data["group_name"].unique():
            group_labels = self.group_data[self.group_data["group_name"] == group_name]["label"]
            group_predictions = self.predictions[self.group_data["group_name"] == group_name]
            accuracy = accuracy_score(group_labels, group_predictions)
            precision = precision_score(group_labels, group_predictions)
            recall = recall_score(group_labels, group_predictions)
            f1 = f1_score(group_labels, group_predictions)
            group_scores[group_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        return group_scores

    def calculate_average_group_score(self):
        """
        Calculate the average score across all groups.

        Returns:
            dict: A dictionary containing the average accuracy, precision, recall, and F1 score.
        """
        group_scores = self.calculate_group_scores()
        average_accuracy = sum(score["accuracy"] for score in group_scores.values()) / len(group_scores)
        average_precision = sum(score["precision"] for score in group_scores.values()) / len(group_scores)
        average_recall = sum(score["recall"] for score in group_scores.values()) / len(group_scores)
        average_f1 = sum(score["f1"] for score in group_scores.values()) / len(group_scores)
        return {
            "accuracy": average_accuracy,
            "precision": average_precision,
            "recall": average_recall,
            "f1": average_f1
        }

    def rank_groups_by_score(self):
        """
        Rank the groups by their scores.

        Returns:
            list: A list of tuples containing the group name and its score, sorted in descending order by score.
        """
        group_scores = self.calculate_group_scores()
        ranked_groups = sorted(group_scores.items(), key=lambda x: x[1]["f1"], reverse=True)
        return ranked_groups