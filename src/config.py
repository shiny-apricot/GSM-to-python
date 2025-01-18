"""
Configuration settings for the GSM Bioinformatics Pipeline.

This module contains all configuration parameters used throughout the pipeline,
organized by processing stage. Each section contains related parameters with
detailed documentation.

Key Configuration Sections:
- Input/Output: Data file paths and formats
- Pipeline Control: High-level pipeline behavior settings
- Model Settings: ML model configuration
- Feature Selection: Parameters for feature filtering and selection
- Data Processing: Data normalization and sampling parameters
- Performance: Execution settings like parallelization
"""

from typing import Literal
from dataclasses import dataclass

# ============================================================================
# Input/Output Configuration
# ============================================================================

INPUT_PATHS = {
    'expression_data': "GDS1962.csv",
    'group_data': "cancer-DisGeNET.txt",
    'output_dir': "output/",
}

GROUP_SETTINGS = {
    'column_name': "diseaseName",
    'min_genes_per_group': 10,
    'max_genes_per_group': 1000,
}

# ============================================================================
# Pipeline Control
# ============================================================================

PIPELINE_SETTINGS = {
    'random_seed': 44,
    'cross_validation_folds': 5,
    'number_of_iterations': 5,
    'save_intermediate_results': True,
}

# ============================================================================
# Model Configuration
# ============================================================================

# Supported model types
ModelType = Literal['DecisionTree', 'RandomForest', 'SVM', 'KNN', 'MLP']

MODEL_SETTINGS = {
    'model_type': 'RandomForest',  # type: ModelType
    'hyperparameters': {
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
        },
        'SVM': {
            'kernel': 'rbf',
            'C': 1.0,
        },
        # Add configurations for other model types as needed
    }
}

# ============================================================================
# Feature Selection Settings
# ============================================================================

FEATURE_SELECTION = {
    'initial_filter_size': 1000,  # 0 to disable initial filtering
    'best_groups_to_keep': 10,
    'min_variance_threshold': 0.01,
    'selection_method': 't_test',  # Options: 't_test', 'f_test', 'mutual_info'
}

# ============================================================================
# Data Processing Settings
# ============================================================================

DATA_PROCESSING = {
    'normalization_method': 'minmax',  # Options: 'minmax', 'zscore', 'robust'
    'train_test_split_ratio': 0.7,
    'class_labels': {
        'positive': "pos",
        'negative': "neg"
    },
    'sampling': {
        'min_class_balance_ratio': 0.5,
        'sampling_method': 'undersampling',  # Options: 'undersampling', 'oversampling', 'hybrid'
    }
}

# ============================================================================
# Performance Settings
# ============================================================================

PERFORMANCE = {
    'num_workers': 1,
    'batch_size': 1000,
    'use_gpu': False,
    'memory_limit': '4GB'
}

# ============================================================================
# Logging Configuration
# ============================================================================

LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'output_file': 'pipeline.log'
}
