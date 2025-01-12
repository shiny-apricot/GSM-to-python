class Scoring:
    """
    A class to score features and groups based on their performance in a model.

    Attributes:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        model (object): The trained model.
        group_data (pd.DataFrame): The group data.
        predictions (pd.Series): The predicted labels.
    """

    def __init__(self, X, y, model, group_data, predictions):
        """
        Initialize the Scoring class.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
            model (object): The trained model.
            group_data (pd.DataFrame): The group data.
            predictions (pd.Series): The predicted labels.
        """
        self.X = X
        self.y = y
        self.model = model
        self.group_data = group_data
        self.predictions = predictions
        
    def run_scoring(self, data, model, feature_group_mapping):
        """
        Run the scoring process.

        Returns:
            tuple: A tuple containing the ranked features, ranked groups, and feature scores.
        """
        ranked_features = self.rank_features_by_score()
        ranked_groups = self.rank_groups_by_score()
        feature_scores = self.calculate_feature_scores()
        return ranked_features, ranked_groups, feature_scores

    def calculate_feature_scores(self):
        """
        Calculate the scores for each feature.

        Returns:
            dict: A dictionary where the keys are feature names and the values are dictionaries containing the accuracy, precision, recall, and F1 score.
        """
        feature_scores = {}
        for feature_name in self.X.columns:
            X_feature = self.X[[feature_name]]
            accuracy = accuracy_score(self.y, self.model.predict(X_feature))
            precision = precision_score(self.y, self.model.predict(X_feature))
            recall = recall_score(self.y, self.model.predict(X_feature))
            f1 = f1_score(self.y, self.model.predict(X_feature))
            feature_scores[feature_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        return feature_scores

    def rank_features_by_score(self):
        """
        Rank the features by their scores.

        Returns:
            list: A list of tuples containing the feature name and its score, sorted in descending order by score.
        """
        feature_scores = self.calculate_feature_scores()
        ranked_features = sorted(feature_scores.items(), key=lambda x: x[1]["f1"], reverse=True)
        return ranked_features

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

    def rank_groups_by_score(self):
        """
        Rank the groups by their scores.

        Returns:
            list: A list of tuples containing the group name and its score, sorted in descending order by score.
        """
        group_scores = self.calculate_group_scores()
        ranked_groups = sorted(group_scores.items(), key=lambda x: x[1]["f1"], reverse=True)
        return ranked_groups

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

