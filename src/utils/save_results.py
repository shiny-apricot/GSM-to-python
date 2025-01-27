"""
🗂️ Result Saving Utility Module

This module provides functionality for saving various types of analysis results
from the GSM pipeline, including:
- DataFrame results (CSV, Excel)
- Model artifacts (pickle)
- Visualization plots
- Analysis metadata (JSON)

Key Classes:
    - ResultSaver: Main class handling save operations

Usage Example:
    saver = ResultSaver(base_dir="results/experiment_1", create_timestamp_subdir=True)
    saver.save_dataframe(df, "filtered_genes", include_timestamp=True)
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Union, Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from joblib import dump
from modeling.run_modeling import ModelingResult
import config

class ResultSaver:
    """Handles saving of various analysis results with error handling and logging."""
    
    def __init__(
        self,
        base_dir: str,
        create_timestamp_subdir: bool = True,
        file_timestamp: bool = False,
        compression: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ResultSaver with parameters.
        
        Args:
            base_dir (str): Base directory for saving results
            create_timestamp_subdir (bool): Whether to create timestamped subdirectory
            file_timestamp (bool): Whether to add timestamps to filenames
            compression (Optional[str]): Compression format for saved files
        """
        self.base_dir = base_dir
        self.create_timestamp_subdir = create_timestamp_subdir
        self.file_timestamp = file_timestamp
        self.compression = compression
        self.logger = logger or logging.getLogger(__name__)
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Create necessary directories for saving results."""
        try:
            if self.create_timestamp_subdir:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.save_dir = Path(self.base_dir) / timestamp
            else:
                self.save_dir = Path(self.base_dir)
            
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Created results directory: {self.save_dir}")
        except Exception as e:
            self.logger.error(f"❌ Failed to create directories: {str(e)}")
            raise

    def _get_filename(self, name: str, extension: str) -> str:
        """Generate filename with optional timestamp."""
        if self.file_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{name}_{timestamp}.{extension}"
        return f"{name}.{extension}"

    def save_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        file_format: str = "csv",
        **kwargs
    ) -> Path:
        """
        Save DataFrame to file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            name (str): Base name for the file
            format (str): Format to save ('csv' or 'excel')
            **kwargs: Additional arguments for pandas save functions
            
        Returns:
            Path: Path to saved file
        """
        try:
            filepath = self.save_dir / self._get_filename(name, file_format)
            
            if file_format == "csv":
                if self.compression:
                    kwargs['compression'] = self.compression
                df.to_csv(filepath, **kwargs)
            elif format == "excel":
                df.to_excel(filepath, **kwargs)
            
            self.logger.info(f"💾 Saved DataFrame to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"❌ Failed to save DataFrame: {str(e)}")
            raise

    def save_plot(
        self,
        name: str,
        dpi: int = 300,
        file_format: str = 'png',
        fig: Optional[Figure] = None,
        **kwargs
    ) -> Path:
        """
        Save current matplotlib figure or specified figure.
        
        Args:
            name (str): Base name for the plot file
            dpi (int): DPI for the saved image
            format (str): Image format ('png', 'pdf', etc.)
            fig (Optional[plt.Figure]): Specific figure to save
            **kwargs: Additional arguments for plt.savefig
        
        Returns:
            Path: Path to saved plot
        """
        filepath = self.save_dir / self._get_filename(name, file_format)
            
        if fig is None:
            fig = plt.gcf()
        
        fig.savefig(str(filepath), dpi=dpi, format=format, **kwargs)
        self.logger.info(f"📊 Saved plot to {filepath}")
        return filepath

    def save_model(self, model: Any, name: str) -> Path:
        """
        Save machine learning model using joblib.
        
        Args:
            model: Trained model to save
            name (str): Base name for the model file
            
        Returns:
            Path: Path to saved model
        """
        try:
            filepath = self.save_dir / self._get_filename(name, "joblib")
            dump(model, filepath)
            self.logger.info(f"🤖 Saved model to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {str(e)}")
            raise

    def save_metadata(self, metadata: Dict[str, Any], name: str) -> Path:
        """
        Save metadata dictionary as JSON.
        
        Args:
            metadata (Dict[str, Any]): Metadata to save
            name (str): Base name for the metadata file
            
        Returns:
            Path: Path to saved metadata
        """
        try:
            filepath = self.save_dir / self._get_filename(name, "json")
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"📝 Saved metadata to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"❌ Failed to save metadata: {str(e)}")
            raise


    def save_modeling_results(self, 
                            results: List[List[ModelingResult]], 
                            experiment_name: str) -> List[List[Dict[str, Path]]]:
        """
        Save all components of modeling results including metrics, groups and features
        into single combined files for each model iteration.
        
        Args:
            results (List[List[ModelingResult]]): Nested list of modeling results to save
            experiment_name (str): Base name of the experiment for file naming
            
        Returns:
            List[List[Dict[str, Path]]]: Nested list of dictionaries containing saved file paths
        """
        # TODO: Save data descriptions and config.py file
        try:
            saved_paths_list = []
            
            for outer_idx, result_group in enumerate(results):
                group_paths = []
                
                for inner_idx, result in enumerate(result_group):
                    saved_paths = {}
                    suffix = f"{experiment_name}_{outer_idx+1}_{inner_idx+1}"
                    
                    # Combine metrics, groups and features into single result data
                    combined_data = {
                        'metrics': asdict(result.metrics),
                        'groups': result.groups,
                        'features': result.features
                    }
                    
                    # # Save combined results
                    # result_path = self.save_metadata(
                    #     combined_data,
                    #     f"{suffix}_results"
                    # )
                    # saved_paths['results'] = result_path
                    
                    # group_paths.append(saved_paths)
                
                # Combine all results for this group into one file
                combined_group_results = {
                    f"group_count_{len(result_group)-i}": {
                        'metrics': asdict(result.metrics),
                        'number_of_groups': len(result.groups),
                        'number_of_features': len(result.features),
                        'groups': result.groups,
                        'features': result.features
                    }
                    for i, result in enumerate(result_group)
                }

                combined_path = self.save_metadata(
                    combined_group_results,
                    f"{experiment_name}_{outer_idx+1}_combined_results"
                )
                
                saved_paths_list.append(group_paths)
            
            return saved_paths_list
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save modeling results: {str(e)}")
            raise
