"""
SplitData Module

Purpose:
    This module defines the SplitData dataclass used for holding split datasets.

Classes:
    SplitData: A dataclass for storing training, validation, and test splits of features and labels.

Usage Example:
    split_data = SplitData(
        X_train=train_features,
        X_val=val_features,
        X_test=test_features,
        y_train=train_labels,
        y_val=val_labels,
        y_test=test_labels
    )
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class TrainTestValSplitData:
    X_train: pd.DataFrame = pd.DataFrame()
    X_val: Optional[pd.DataFrame] = None
    X_test: pd.DataFrame = pd.DataFrame()
    y_train: pd.Series = pd.Series()
    y_val: Optional[pd.Series] = None
    y_test: pd.Series = pd.Series()