import pandas as pd


class Modeling:
    """
    This class handles the modeling process.

    Attributes:
        model (object): The machine learning model to use.
        data_preprocessor (DataPreprocessor): The data preprocessor.
        feature_scorer (FeatureScorer): The feature scorer.
        group_scorer (GroupScorer): The group scorer.
    """
    def __init__(self, model, data_preprocessor, feature_scorer, group_scorer):
        """
        Initialize the Modeling class.

        Args:
            model (object): The machine learning model to use.
            data_preprocessor (DataPreprocessor): The data preprocessor.
            feature_scorer (FeatureScorer): The feature scorer.
            group_scorer (GroupScorer): The group scorer.
        """
        self.model = model
        self.data_preprocessor = data_preprocessor
        self.feature_scorer = feature_scorer
        self.group_scorer = group_scorer

    def run(self, 
            data_x: pd.DataFrame,
            data_y: pd.Series,
            target_column: str,
            feature_column: str,
            feature_ranks: pd.Series,
            group_ranks: pd.Series):
        """
        Run the modeling process.

        Args:
            data (pd.DataFrame): The dataset to use.
            target_column (str): The name of the target column in the dataset.
            feature_column (str): The name of the feature column in the dataset.
            group_column (str): The name of the group column in the dataset.
            feature_ranks (pd.Series): The feature ranks.
            group_ranks (pd.Series): The group ranks.

        Returns:
            tuple: The trained model, the feature ranks, and the group ranks.
        """
        # Preprocess the data
        preprocessed_data = self.data_preprocessor.preprocess_data(data, target_column, feature_column, group_column)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            preprocessed_data[feature_column],
            preprocessed_data[target_column],
            test_size=0.2,
            random_state=42
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate the feature scores
        feature_scores = self.feature_scorer.calculate_scores(y_test, y_pred, y_pred_proba)

        # Calculate the group scores
        group_scores = self.group_scorer.calculate_scores(y_test, y_pred, y_pred_proba)

        # Update the feature ranks
        feature_ranks = pd.concat([feature_ranks, feature_scores], axis=0).groupby("feature").mean()

        # Update the group ranks
        group_ranks = pd.concat([group_ranks, group_scores], axis=0).groupby("group").mean()

        return self.model, feature_ranks, group_ranks
