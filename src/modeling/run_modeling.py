from typing import Tuple, List, Optional
from numpy import number
import pandas as pd
import logging
from sklearn.base import BaseEstimator
from dataclasses import dataclass

from scoring.metrics import MetricsData
from grouping.group_feature_mapping import GroupFeatureMappingData
from .evaluator import calculate_metrics, update_feature_ranks
from .modeling_utils import preprocess_features, filter_features_of_groups
from machine_learning.classification import train_model, predict
from scoring.metrics import MetricsData 


@dataclass
class ModelingResult:
    trained_model: Optional[BaseEstimator]
    metrics: MetricsData
    groups: List[str]
    features: List[str]

def run_modeling(
    data_train_x: pd.DataFrame,
    data_train_y: pd.Series,
    data_test_x: pd.DataFrame,
    data_test_y: pd.Series,
    model_name: str,
    group_ranks: List[MetricsData],
    group_feature_mapping: List[GroupFeatureMappingData],
    logger: logging.Logger
) -> ModelingResult:
    """
    Run the complete modeling pipeline.
    
    Args:
        data_train_x (pd.DataFrame): Training feature DataFrame
        data_train_y (pd.Series): Training target Series
        data_test_x (pd.DataFrame): Testing feature DataFrame
        data_test_y (pd.Series): Testing target Series
        model_name (str): Machine learning model to use
        group_ranks (List[MetricsData]): Current group rankings
        logger (logging.Logger): Logger for logging information
        
    Returns:
        ModelingResult: Dataclass containing trained model, performance metrics, 
                        updated feature rankings, and updated group rankings
    """
    # Preprocess features
    processed_x = preprocess_features(data_train_x.copy(), logger=logger)
    
    # pick the best 10 groups
    group_list = group_ranks[:10]

    # Convert MetricsData objects to group name strings
    group_name_list = [group.name for group in group_list if group.name is not None]

    # Filter features based on best groups
    filtered_features = filter_features_of_groups(group_list=group_name_list,
                                           group_feature_mapping=group_feature_mapping,
                                           data_x=processed_x,
                                           logger=logger)
    

    # Check which features exist in the data
    available_features = [f for f in filtered_features if f in data_train_x.columns]
    
    # if len(available_features) < len(filtered_features):
    #     logger.warning(f"Some features were not found in the data. Expected {len(filtered_features)}, found {len(available_features)}")
    
    if not available_features:
        raise ValueError("No valid features found in the data")
    
    # Train model with available features
    trained_model = train_model(model_name=model_name, 
                              data_x=data_train_x[available_features], 
                              data_y=data_train_y)

    # Evaluate model
    predictions = predict(trained_model, data_test_x[available_features])
    metrics = calculate_metrics(data_test_y, 
                                predictions, 
                                logger=logger)

    # Update feature ranks based on model performance to be used in next iteration
    return ModelingResult(trained_model=trained_model,
                            metrics=metrics,
                            groups=group_name_list,
                            features=filtered_features)

