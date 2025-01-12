# feature_scorer.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scoring.group_scorer import GroupScorer

class FeatureScorer:
    """
    A class to score features based on their performance in a model.

    Attributes:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        model (object): The trained model.
    """

    def __init__(self, X, y, model):
        """
        Initialize the FeatureScorer class.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
            model (object): The trained model.
        """
        self.X = X
        self.y = y
        self.model = model

    def calculate_feature_scores(self):
        pass

    def rank_features_by_score(self):
        pass
    
    def rank_groups_by_score(self):
        """
        Rank the groups by their scores.

        Returns:
            list: A list of tuples containing the group name and its score, sorted in descending order by score.
        """
        pass
    
    def calculate_group_scores(self):
        pass