"""
Module for scoring gene groups in the GSM pipeline.

Functions:
- score_groups: Calculate performance metrics for gene groups
"""

from typing import Dict
import numpy as np
import pandas as pd
from .metrics import MetricsData
from sklearn.model_selection import cross_val_score, cross_validate
from machine_learning.classification import train_model, predict, get_classifier_object
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

def score_data(data_x: pd.DataFrame,
               labels: pd.Series, 
               classifier_name: str) -> MetricsData:
    """
    Calculate performance score for a dataframe using cross-validation.

    Args:
        data_x (pd.DataFrame): DataFrame containing the features.
        labels (pd.Series): Series containing the labels.
        classifier_name (str): Name of the classifier to use.
        evaluation_metric (str): Metric to evaluate model ('accuracy', 'f1', 'precision', 'recall').

    Returns:
        MetricsData: Dataclass containing the performance metrics.

    Raises:
        ValueError: If invalid evaluation metric is provided.
    """

    # Get classifier object
    classifier = get_classifier_object(classifier_name)

    # Modify the scoring metrics to include zero_division parameter
    scoring_metrics = {
        'accuracy': 'accuracy',
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=1),
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=1),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=1)
    }

    # Perform cross-validation for all metrics at once
    scores = cross_validate(
        classifier,
        data_x,
        labels,
        cv=5,
        scoring=scoring_metrics,
        return_train_score=False,
    )

    # Calculate means and standard deviations for all metrics
    metrics = MetricsData(
        accuracy=float(np.mean(scores['test_accuracy'])),
        accuracy_std=float(np.std(scores['test_accuracy'])),
        f1=float(np.mean(scores['test_f1_macro'])),
        f1_std=float(np.std(scores['test_f1_macro'])),
        precision=float(np.mean(scores['test_precision_macro'])),
        precision_std=float(np.std(scores['test_precision_macro'])),
        recall=float(np.mean(scores['test_recall_macro'])),
        recall_std=float(np.std(scores['test_recall_macro']))
    )

    return metrics