# model_evaluator.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class ModelEvaluator:
    def __init__(self, model, data, target_column, test_size=0.2, random_state=42):
        """
        Initialize the ModelEvaluator class.

        Args:
            model (object): The machine learning model to evaluate.
            data (pd.DataFrame): The dataset to use for evaluation.
            target_column (str): The name of the target column in the dataset.
            test_size (float, optional): The proportion of the dataset to use for testing. Defaults to 0.2.
            random_state (int, optional): The random seed to use for splitting the data. Defaults to 42.
        """
        self.model = model
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        """
        Split the dataset into training and testing sets.

        Returns:
            tuple: The training data, testing data, training target, and testing target.
        """
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate the performance of the model on the testing data.

        Args:
            X_train (pd.DataFrame): The training data.
            X_test (pd.DataFrame): The testing data.
            y_train (pd.Series): The training target.
            y_test (pd.Series): The testing target.

        Returns:
            dict: A dictionary containing the accuracy, precision, recall, and F1 score of the model.
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def run_evaluation(self):
        """
        Run the evaluation process.

        Returns:
            dict: A dictionary containing the accuracy, precision, recall, and F1 score of the model.
        """
        X_train, X_test, y_train, y_test = self.split_data()
        metrics = self.evaluate_model(X_train, X_test, y_train, y_test)
        return metrics