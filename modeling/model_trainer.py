# model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self, model, data, target_column, test_size=0.2, random_state=42):
        """
        Initialize the ModelTrainer class.

        Args:
            model (object): The machine learning model to train.
            data (pd.DataFrame): The dataset to use for training.
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

    def train_model(self, X_train, y_train):
        """
        Train the model on the training data.

        Args:
            X_train (pd.DataFrame): The training data.
            y_train (pd.Series): The training target.

        Returns:
            object: The trained model.
        """
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_trained_model(self, X_test, y_test):
        """
        Evaluate the performance of the trained model on the testing data.

        Args:
            X_test (pd.DataFrame): The testing data.
            y_test (pd.Series): The testing target.

        Returns:
            dict: A dictionary containing the accuracy, precision, recall, and F1 score of the model.
        """
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

    def run_training(self):
        """
        Run the training process.

        Returns:
            tuple: The trained model and a dictionary containing the accuracy, precision, recall, and F1 score of the model.
        """
        X_train, X_test, y_train, y_test = self.split_data()
        trained_model = self.train_model(X_train, y_train)
        metrics = self.evaluate_trained_model(X_test, y_test)
        return trained_model, metrics