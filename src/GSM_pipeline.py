from utils import logger
from config import (INPUT_EXPRESSION_DATA, INPUT_GROUP_DATA, OUTPUT_DIR, 
                     RANDOM_SEED, CROSS_VALIDATION_FOLDS, NUMBER_OF_ITERATIONS, 
                     SAVE_INTERMEDIATE_RESULTS, TRAIN_TEST_SPLIT_RATIO, MODEL_NAME, LABEL_COLUMN_NAME, NORMALIZATION_METHOD,
                     CLASS_LABELS_NEGATIVE, CLASS_LABELS_POSITIVE)

import pandas as pd

from grouping.run_grouping import run_grouping
from scoring.run_scoring import run_scoring
from modeling.run_modeling import run_modeling
from data_processing.data_loader import load_input_file, load_group_file
from data_processing.data_preprocess import preprocess_data, preprocess_grouping_data
from data_processing.normalization import normalize_data
from data_processing.train_test_splitter import split_data
from data_processing.preliminary_filtering import preliminary_filter


def gsm_run(input_data: pd.DataFrame,
            group_data: pd.DataFrame,
            sample_ratio: float = TRAIN_TEST_SPLIT_RATIO,
            n_iteration_workflow: int = NUMBER_OF_ITERATIONS,
            model_name: str = MODEL_NAME,
            label_column_name: str = LABEL_COLUMN_NAME,
            normalization_method: str = NORMALIZATION_METHOD,
            notebook_mode: bool = False,
            ) -> None:
    
    logger = logger.setup_logger()
    
    logger.info("Starting GSM pipeline...")
    
    # Run the GSM pipeline
    logger.info("Start data preprocessing.")
    data_preprocessed_train, data_preprocessed_test = preprocess_data(input_data,
                                        label_column_name,
                                        logger=logger,
                                        label_of_negative_class=CLASS_LABELS_NEGATIVE,
                                        label_of_positive_class=CLASS_LABELS_POSITIVE,
                                        normalization_method=normalization_method,
                                        )
    logger.info("Data preprocessing completed.")
    
    group_data_processed = preprocess_grouping_data(group_data, logger=logger)
    logger.info("Grouping data preprocessing completed.")
    
    for i in range(n_iteration_workflow):    
        logger.info(f"Iteration {i} for gsm_main_loop started...")
        gsm_main_loop(data_preprocessed_train, group_data_processed, model_name, logger)
        logger.info(f"Iteration {i} for gsm_main_loop completed.")

    logger.info("GSM pipeline completed successfully.")

def gsm_main_loop(data: pd.DataFrame, 
                  grouping_data: pd.DataFrame,
                  model_name: str, 
                  logger,
                  ) -> None:
    logger.info("Splitting data into training and testing sets...")
    data_train_x, data_test_x, data_train_y, data_test_y = split_data(data, LABEL_COLUMN_NAME, test_size=0.2, stratify=True)
    logger.info("Data split completed.")
    # Apply preliminary filter to features
    logger.info("Applying preliminary filter to training data...")
    filtered_train = preliminary_filter(data_train_x, data_train_y, logger=logger)
    logger.info("Preliminary filtering completed.")

    # Grouping
    logger.info("Running grouping...")
    grouping_result_object = run_grouping(filtered_train, grouping_data)
    logger.info("Grouping completed.")

    # Scoring
    logger.info("Running scoring...")
    ranked_features, ranked_groups, feature_scores = run_scoring(data_train_x, 
                                                                 model_name, 
                                                                 grouping_result_object.grouped_data, 
                                                                 data_train_y)
    logger.info("Scoring completed.")

    # Modeling
    logger.info("Running modeling...")
    trained_model, metrics, feature_ranks, group_ranks = run_modeling(data_filtered_x, 
                                                                      data_filtered_y, 
                                                                      model_name, 
                                                                      feature_ranks=ranked_features, 
                                                                      group_ranks=ranked_groups)
    logger.info("Modeling completed successfully.")