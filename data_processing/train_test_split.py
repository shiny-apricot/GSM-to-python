
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to split.
        target_column (str): The name of the target column in the dataset.
        test_size (float, optional): The proportion of the dataset to use for testing. Defaults to 0.2.
        random_state (int, optional): The random seed to use for splitting the data. Defaults to 42.

    Returns:
        tuple: The training data, testing data, training target, and testing target.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test