

from utils.logger import Logger
from grouping.input_loader import load_input_file, load_group_file
from scoring.metrics import calculate_metrics
from data_processing.DataPreprocessor import DataPreprocessor

import pandas as pd
import sys
import pathlib
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Sequence, Iterable, Iterator, Set

from grouping.Grouping import Grouping
from scoring.Scoring import Scoring
from modeling.Modeling import Modeling

class GSM_Pipeline:
    def __init__(self, 
                 input_file, 
                 group_file, 
                 data_preprocessor: DataPreprocessor, 
                 model,
                 logger: Logger,
                 project_folder: pathlib.Path,
                 output_folder: pathlib.Path,
                 notebook_mode: bool = False,
                 ):
        # Initialize instance variables
        self.input_file = input_file
        self.group_file = group_file
        self.model = model
        self.data_preprocessor = data_preprocessor
        self.logger = logger
        self.notebook_mode = notebook_mode
        self.project_folder = project_folder
        self.output_folder = output_folder
        
        if self.logger is None:
            self.logger = Logger("GSM_Pipeline")
        
        self.data = None
        self.groups = None
        pass
    
    def gsm_run(self, 
                data: pd.DataFrame, 
                sample_ratio: float, 
                n_iteration_workflow: int, 
                model, 
                grouper: Grouping, 
                scorer: Scoring, 
                modeler: Modeling,
                data_preprocessor: DataPreprocessor,
                ):
        # Run the GSM pipeline
        data_checked = self.preliminary_checks(data)
        data_loaded = self.load_data(data_checked)
        data_preprocessed = self.preprocess_data(data_preprocessor=data_preprocessor, 
                                                 data=data_loaded)
        data_normalized = self.normalize_data(data_preprocessed)
        
        for i in range(n_iteration_workflow):    
            self.logger.info(f"Iteration {i} for gsm_main_loop started...")
            self.gsm_main_loop(
                data_normalized,
                model,
                sample_ratio,
                grouper,
                scorer,
                modeler,
            )
            self.logger.info(f"Iteration {i} for gsm_main_loop completed.")

        self.write_output()
        self.visualize_results()

        self.logger.info("GSM pipeline completed successfully.")
        pass
        
    def gsm_main_loop(self, 
                      data, 
                      model, 
                      sample_ratio, 
                      grouper: Grouping, 
                      scorer: Scoring, 
                      modeler: Modeling):
        # Perform the main loop of the GSM pipeline
        data_sampled = self.sample_by_ratio(sample_ratio, data)
        # Split the data
        data_train, data_test = self.train_test_split(data_sampled)
        # Filter unimportant features
        data_filtered_x, data_filtered_y = self.filter_unimportant_features(data_train, data_test)
        # Apply grouping scoring modeling
        self.apply_grouping_scoring_modeling(model, 
                                             data_filtered_x,
                                             data_filtered_y,
                                             self.groups, 
                                             grouper, 
                                             scorer, 
                                             modeler)
        pass
    
    def preliminary_checks(self, data) -> pd.DataFrame:
        # Perform preliminary checks on the files, and the project environment
        return pd.DataFrame()

    def load_data(self, data):
        # Load data from input file
        self.data = load_input_file(self.project_folder, self.input_file)
        self.groups = load_group_file(self.project_folder, self.group_file)
        pass

    def preprocess_data(self, 
                        data_preprocessor: DataPreprocessor, 
                        data
                        ) -> pd.DataFrame:
        # Preprocess data (e.g., handle missing values, normalize)
        self.preprocessed_data = data_preprocessor.convert_labels_to_binary(data)
        return self.preprocessed_data
    
    def normalize_data(self, data):
        # Normalize data
        self.normalized_data = self.data_preprocessor.normalize_data(data)
        return self.normalized_data

    def evaluate_model(self, model, data):
        # Evaluate the trained model
        pass
    
    def sample_by_ratio(self, data, sample_ratio) -> pd.DataFrame:
        # Sample data based on a specified ratio
        return pd.DataFrame()
    
    def train_test_split(self, data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Split data into training and testing sets
        train_data, test_data = self.data_preprocessor.train_test_split(data)
        return train_data, test_data
    
    def filter_unimportant_features(self, data_train, data_test):
        # Select relevant features
        return pd.DataFrame(), pd.DataFrame()
    
    def apply_grouping_scoring_modeling(self, 
                                        model, 
                                        data_x,
                                        data_y, 
                                        group_data, 
                                        grouper: Grouping,
                                        scorer: Scoring,
                                        modeler: Modeling
                                        ):
        label_data = data_x["label"]
        feature_group_mapping = grouper.run_grouping(group_data, data_x, label_data)
        predictions = scorer.run_scoring(data_x, model, feature_group_mapping)
        modeler.run(data_x=data_x,
                    data_y=data_y,
                    target_column="label",
                    feature_column="feature",
                    feature_ranks=feature_group_mapping,
                    group_ranks=predictions)

        pass
    
    def write_output(self):
        pass
    
    def visualize_results(self):
        pass


