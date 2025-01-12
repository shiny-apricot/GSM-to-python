from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import pandas as pd

def select_k_best(X, y, k=10):
    """
    Select the top k features based on the chi-squared statistic.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        k (int): The number of features to select.

    Returns:
        pd.DataFrame: The selected feature data.
    """
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the selected feature names
    selected_feature_names = X.columns[selector.get_support()]

    # Convert the selected features to a DataFrame
    X_selected = pd.DataFrame(X_new, columns=selected_feature_names)

    return X_selected