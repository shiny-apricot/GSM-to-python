# scoring/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate the metrics for a given set of true labels and predicted labels.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        y_pred_proba (np.ndarray, optional): The predicted probabilities. Defaults to None.

    Returns:
        dict: A dictionary containing the accuracy, precision, recall, F1 score, and AUC-ROC score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

    if y_pred_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)

    return metrics

def calculate_group_metrics(y_true, y_pred, group_labels):
    """
    Calculate the metrics for each group in a given set of true labels and predicted labels.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        group_labels (np.ndarray): The group labels.

    Returns:
        dict: A dictionary where the keys are group labels and the values are dictionaries containing the accuracy, precision, recall, F1 score, and AUC-ROC score.
    """
    group_metrics = {}
    for group_label in np.unique(group_labels):
        group_mask = group_labels == group_label
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]

        group_metrics[group_label] = calculate_metrics(group_y_true, group_y_pred)

    return group_metrics