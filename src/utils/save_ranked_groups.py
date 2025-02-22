"""
📊 Utility module for saving ranked group results to Excel files
Key functions:
    - save_ranked_groups: Saves group rankings with metrics to Excel (ranks calculated automatically)
"""

from dataclasses import dataclass
from typing import List
import pandas as pd
import logging
from pathlib import Path

from scoring.metrics import MetricsData


def save_ranked_groups(
    output_path: str,
    group_list: List[MetricsData],
    iteration: int,
    logger: logging.Logger
) -> None:
    """
    Saves ranked groups and their metrics to an Excel file.
    Ranks are calculated automatically based on accuracies.
    
    Args:
        output_path: Path to save Excel file
        group_list: List of MetricsData objects containing group metrics
        iteration: Current iteration number
        logger: Logger instance
    """
    try:
        logger.info("🔄 Starting to save ranked groups to Excel...")
        
        df = pd.DataFrame({
            'Group Name': [m.name for m in group_list],
            'Accuracy': [m.accuracy for m in group_list],
            'Iteration': [iteration] * len(group_list)
        })
        
        df = df.sort_values('Accuracy', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'Group Name', 'Accuracy', 'Iteration']]
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and load existing data
        if Path(output_path).exists():
            existing_df = pd.read_excel(output_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_excel(output_path, index=False)
        logger.info(f"✅ Successfully saved ranked groups to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error saving ranked groups: {str(e)}")
        raise
