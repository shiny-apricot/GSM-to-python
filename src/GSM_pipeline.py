
# TODO: Implement unit tests for gsm_run and gsm_main_loop functions
# TODO: Optimize data preprocessing steps for large datasets using dask
# TODO: Refactor gsm_main_loop to reduce complexity and improve readability
# TODO: Add support for additional ML models in the modeling stage
# TODO: Add functionality to visualize intermediate results and final outputs
# TODO: Document each function with comprehensive docstrings and usage examples

"""
🧬 GSM_pipeline.py - Main Pipeline Implementation for Gene Analysis

Purpose:
    Core implementation of the Grouping-Scoring-Modeling (GSM) pipeline for gene analysis.
    This pipeline processes gene expression data through multiple stages to identify
    significant gene groups and build predictive models.

Key Components:
    1. Data Preprocessing: Normalizes and validates input data
    2. Grouping: Groups genes based on biological relationships
    3. Scoring: Evaluates gene groups using ML metrics
    4. Modeling: Trains and validates ML models on selected groups

Key Functions:
    📊 gsm_run: Main entry point - orchestrates the complete pipeline execution
    🔄 gsm_main_loop: Core processing loop implementing the GSM workflow stages

Usage Example:
    >>> # Load input data
    >>> expression_data = pd.read_csv("expression_data.csv")
    >>> group_data = pd.read_csv("group_data.csv")
    >>> logger = setup_logger()
    >>> 
    >>> # Configure and run pipeline
    >>> config = GSMConfig(sample_ratio=0.8, n_iteration_workflow=5)
    >>> gsm_run(expression_data, group_data, logger, config)

Notes:
    - Ensures reproducibility through fixed random seeds
    - Implements comprehensive error handling and logging
    - Supports both notebook and script execution modes
"""

##### Imports #####
from math import exp
from config import (INPUT_EXPRESSION_DATA, INPUT_GROUP_DATA, OUTPUT_DIR, 
                     RANDOM_SEED, CROSS_VALIDATION_FOLDS, NUMBER_OF_ITERATIONS, 
                     SAVE_INTERMEDIATE_RESULTS, TRAIN_TEST_SPLIT_RATIO, MODEL_NAME, LABEL_COLUMN_NAME, NORMALIZATION_METHOD,
                     CLASS_LABELS_NEGATIVE, CLASS_LABELS_POSITIVE, BEST_GROUPS_TO_KEEP)

import pandas as pd
from dataclasses import dataclass
from typing import List

from grouping import group_feature_mapping
from grouping.grouping_utils import create_group_feature_mapping
from grouping.run_grouping import run_grouping
from scoring.run_scoring import run_scoring
from modeling.run_modeling import ModelingResult, run_modeling
from data_processing.data_loader import load_input_file, load_group_file
from data_processing.data_preprocess import preprocess_data, preprocess_grouping_data
from data_processing.normalization import normalize_data
from data_processing.train_test_splitter import split_data
from data_processing.preliminary_filtering import preliminary_ttest_filter
from data_processing import TrainTestValSplitData
from utils import save_results
from grouping.grouping_utils import create_group_feature_mapping

##### Data Structures #####
@dataclass
class GSMConfig:
    """Configuration parameters for the GSM pipeline.
    
    Attributes:
        sample_ratio (float): Train/test split ratio (default: from config)
        n_iteration_workflow (int): Number of GSM workflow iterations
        model_name (str): ML model identifier
        label_column_name (str): Column name containing class labels
        normalization_method (str): Data normalization strategy
    """
    sample_ratio: float = TRAIN_TEST_SPLIT_RATIO
    n_iteration_workflow: int = NUMBER_OF_ITERATIONS
    model_name: str = MODEL_NAME
    label_column_name: str = LABEL_COLUMN_NAME
    normalization_method: str = NORMALIZATION_METHOD

##### Main Pipeline Functions #####
def gsm_run(input_data: pd.DataFrame,
            group_data: pd.DataFrame,
            logger,
            config: GSMConfig = GSMConfig(),
            notebook_mode: bool = False) -> None:
    """
    Main entry point for the GSM pipeline execution.

    Workflow:
    1. Data preprocessing and normalization
    2. Group data preparation
    3. Iterative GSM workflow execution
    
    Args:
        input_data: Gene expression matrix (samples × genes)
        group_data: Gene grouping information
        logger: Logging interface for pipeline tracking
        config: Pipeline configuration parameters
        notebook_mode: Enable notebook-specific optimizations

    Notes:
        - Implements comprehensive error logging
        - Supports intermediate result saving
        - Handles both positive/negative class labels
    """
    logger = logger.setup_logger()
    logger.info("🚀 Starting GSM pipeline...")

    # Run the GSM pipeline
    logger.info("Start data preprocessing.")
    data_preprocessed_train, data_preprocessed_test = preprocess_data(input_data,
                                        config.label_column_name,
                                        logger=logger,
                                        label_of_negative_class=CLASS_LABELS_NEGATIVE,
                                        label_of_positive_class=CLASS_LABELS_POSITIVE,
                                        normalization_method=config.normalization_method)
    logger.info("Data preprocessing completed.")
    
    group_data_processed = preprocess_grouping_data(group_data, logger=logger)
    logger.info("Grouping data preprocessing completed.")
    
    modeling_result_list : List[List[ModelingResult]] = []
    for i in range(config.n_iteration_workflow):    
        logger.info(f"Iteration {i} for gsm_main_loop started...")
        modeling_result = gsm_main_loop(data_preprocessed_train, group_data_processed, config.model_name, logger)
        modeling_result_list.append(modeling_result)
        logger.info(f"Iteration {i} for gsm_main_loop completed.")

    # Save intermediate results if enabled
    if SAVE_INTERMEDIATE_RESULTS:
        logger.info("Saving intermediate results...")
        result_saver = save_results.ResultSaver(base_dir=str(OUTPUT_DIR), 
                                                create_timestamp_subdir=True,
                                                logger=logger)
        result_saver.save_modeling_results(modeling_result_list, "modeling_results")
        logger.info("Intermediate results saved.")
    logger.info("GSM pipeline completed successfully.")

def gsm_main_loop(data: pd.DataFrame, 
                  grouping_data: pd.DataFrame,
                  model_name: str, 
                  logger) -> List[ModelingResult]:
    """
    Executes one complete iteration of the GSM workflow.

    Pipeline Stages:
    1. Data splitting (train/test)
    2. Preliminary feature filtering (t-test)
    3. Gene grouping analysis
    4. Group performance scoring
    5. Model training and evaluation

    Args:
        data: Preprocessed expression data
        grouping_data: Processed group definitions
        model_name: Selected ML model identifier
        logger: Pipeline logging interface

    Technical Notes:
        - Uses stratified sampling for data splitting
        - Implements vectorized operations for performance
        - Supports intermediate result caching
    """
    logger.info("##### Starting GSM Main Loop #####")
    
    # Data Splitting
    logger.info("📊 Splitting data into training and testing sets...")
    train_test_split_data = split_data(data, LABEL_COLUMN_NAME, test_size=0.2, stratify=True)
    logger.info("Data split completed.")
    
    # Feature Filtering
    logger.info("🔍 Applying preliminary t-test filter...")
    filtered_train = preliminary_ttest_filter(train_test_split_data.X_train, 
                                        train_test_split_data.y_train,
                                        logger=logger)
    logger.info("Preliminary filtering completed.")

    # Gene Grouping
    logger.info("🔗 Running gene grouping analysis...")
    group_feature_mapping_data = create_group_feature_mapping(grouping_data)
    grouping_result_object = run_grouping(grouping_data, 
                                          group_feature_mappings= group_feature_mapping_data,
                                          logger=logger)
    logger.info("Grouping completed.")

    # Group Scoring
    logger.info("📈 Evaluating group performance...")
    # Get the ranked groups based on F1 score in descending order
    ranked_groups= run_scoring(data_x=train_test_split_data.X_train, 
                               labels=train_test_split_data.y_train,
                               model_name=model_name, 
                               groups=grouping_result_object,
                               logger=logger)
    logger.info("Scoring completed.")

    # Model Training
    logger.info("🤖 Training and evaluating models...")
    # Start by using BEST_GROUPS_TO_KEEP number of groups,
    # Then decrease the number of groups iteratively
    modeling_result_list = []
    for i in range(BEST_GROUPS_TO_KEEP, 0, -1):
        logger.info(f"Training model with top {i} groups...")
        modeling_result = run_modeling(data_train_x=train_test_split_data.X_train, 
                                       data_train_y=train_test_split_data.y_train,
                                       data_test_x=train_test_split_data.X_test,
                                       data_test_y=train_test_split_data.y_test,
                                       group_ranks=ranked_groups[:i],
                                       group_feature_mapping=group_feature_mapping_data,
                                       model_name=model_name, 
                                       logger=logger)
        modeling_result_list.append(modeling_result)
        logger.info(f"Modeling with top {i} groups completed")

    logger.info("Modeling completed successfully.")
    return modeling_result_list
