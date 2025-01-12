# data_processing/normalization.py

import pandas as pd

def normalize_data(data, normalization_method='minmax'):
    """
    Normalize the data using the specified method.

    Args:
        data (pd.DataFrame): The data to normalize.
        normalization_method (str, optional): The normalization method to use. Defaults to 'minmax'.

    Returns:
        pd.DataFrame: The normalized data.
    """
    if normalization_method == 'minmax':
        # Min-Max Scaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    elif normalization_method == 'zscore':
        # Standard Scaler (Z-Score)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    else:
        raise ValueError("Invalid normalization method. Please choose from 'minmax' or 'zscore'.")

    return normalized_data