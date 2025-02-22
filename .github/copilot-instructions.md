# GSM Bioinformatics Pipeline Development Guide 🧬

## Purpose
This guide outlines the development standards for the Grouping-Scoring-Modeling (GSM) pipeline, designed for researchers with varying Python expertise levels.

## Pipeline Overview
```
Data → Filter → Group → Score → Model → Predict
```

Each stage processes gene data sequentially, transforming raw input into actionable predictions.

## Core Development Principles

### 1. Code Organization 🏗️
- Write modular, single-responsibility components
- Follow top-down development approach:
  1. Start with main pipeline function (gsm_pipeline.py)
  2. Implement major stage functions (filter.py, group.py, score.py, model.py)
  3. Break down each stage into smaller helper functions
  4. Create utility functions last
- Prefer pure functions over classes
- Use dataclasses for ALL data structures
- NEVER use dictionaries, tuples, or named tuples - always use dataclasses instead

Example of top-down hierarchy:
```python
# Level 1: Main Pipeline (gsm_pipeline.py)
def gsm_pipeline(input_data: GeneData, config: Config, logger: Logger) -> Results:
    """Main pipeline orchestrating the entire GSM analysis."""
    filtered_data = filter_genes(input_data, config, logger)
    groups = group_genes(filtered_data, config, logger)
    scores = score_groups(groups, config, logger)
    return train_model(scores, config, logger)

# Level 2: Major Stage (group.py)
def group_genes(filtered_data: FilteredData, config: Config, logger: Logger) -> list[GeneGroup]:
    """Major stage function for gene grouping."""
    normalized_data = normalize_expression(filtered_data)
    clusters = perform_clustering(normalized_data)
    return create_gene_groups(clusters)

# Level 3: Helper Functions (grouping/cluster_utils.py)
def perform_clustering(normalized_data: np.ndarray) -> np.ndarray:
    """Helper function for specific clustering logic."""
    # Implementation details
```

```python
from dataclasses import dataclass

@dataclass
class GeneGroup:
    name: str
    genes: list[str]
    score: float = 0.0
```

### 2. Code Clarity for Non-Python Experts
- Write self-documenting code
- Include detailed comments in plain English
- Add visual separators for code sections
```python
##### DATA PREPROCESSING #####
def preprocess_gene_data(data: GeneData, logger: Logger) -> ProcessedData:
    """Transform raw gene data into analysis-ready format.
    
    Example:
        raw_data = load_data("genes.csv")
        processed = preprocess_gene_data(raw_data, logger)
    """
    logger.info("🔄 Starting data preprocessing...")
```

### 3. Function Design Rules
- One function = one task
- Maximum 20 lines per function
- Maximum 2 levels of nesting
- Always use type hints
```python
def calculate_gene_score(
    gene_id: str,
    expression_data: np.ndarray,
    *,  # Force keyword arguments
    threshold: float = 0.05,
    logger: Logger
) -> float:
```

### 4. Error Handling & Logging
- Log all critical operations
- Use custom exceptions for domain-specific errors
```python
class GeneAnalysisError(Exception):
    """Base exception for gene analysis errors."""
    pass

def analyze_genes(genes: list[str], logger: Logger) -> GeneResults:
    try:
        logger.info(f"📊 Analyzing {len(genes)} genes...")
        # Analysis code here
    except ValueError as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise GeneAnalysisError(f"Gene analysis failed: {str(e)}")
```

### 5. Performance Optimization
- Use numpy for numerical operations
- Implement dask for large datasets
- Profile code regularly
```python
import dask.dataframe as dd

def process_large_dataset(file_path: str) -> dd.DataFrame:
    """Process gene expression data using dask for memory efficiency."""
    return dd.read_csv(file_path).map_partitions(process_partition)
```

## Project Structure
```
gsm_pipeline/
├── src/
│   ├── filtering/       # T-test based filtering
│   ├── grouping/        # Gene group analysis
│   ├── scoring/         # ML-based scoring
│   ├── modeling/        # Model training
│   ├── ml_utils/        # ML helpers
│   └── utils/           # Common utilities
├── data/
│   ├── raw/            # Input data
│   └── processed/      # Analysis results
└── tests/              # Test suites
```

## Documentation Standards

### File Headers
```python
"""
Gene Group Analysis Module 🧬

Purpose:
    Implements gene grouping algorithms based on expression patterns.

Key Functions:
    - group_genes(): Creates gene groups from expression data
    - score_groups(): Evaluates group significance
    - optimize_groups(): Refines group assignments

Example Usage:
    groups = group_genes(expression_data, logger)
    scores = score_groups(groups, logger)
"""
```

### Function Documentation
```python
def group_genes(
    expression_data: np.ndarray,
    *,
    min_size: int = 10,
    logger: Logger
) -> list[GeneGroup]:
    """Group genes based on expression patterns.

    Args:
        expression_data: Gene expression matrix (genes × samples)
        min_size: Minimum genes per group (default: 10)
        logger: Logger instance for tracking

    Returns:
        List of GeneGroup objects

    Example:
        >>> data = load_expression_data("data.csv")
        >>> groups = group_genes(data, min_size=15, logger=logger)
        >>> print(f"Found {len(groups)} gene groups")
    """
```

## Testing Requirements
- Write tests for all core functions
- Include edge cases
- Test with small datasets first
```python
def test_gene_grouping():
    """Test gene grouping with a small dataset."""
    test_data = load_test_data()
    groups = group_genes(test_data, min_size=5, logger=test_logger)
    assert len(groups) > 0, "Should create at least one group"
    assert all(len(g.genes) >= 5 for g in groups), "Groups should meet size requirement"
```

## Dependencies
- pandas
- numpy
- scikit-learn
- dask
- matplotlib
- seaborn