# G-S-M Bioinformatics Data Pipeline Project

## Purpose
Implementation of Grouping-Scoring-Modeling (GSM) pipeline for gene analysis.

## Project Overview
This project implements a modular data pipeline for bioinformatics analysis using the GSM approach:
1. **Load**: Load and validate input data
2. **Filter**: Preliminary gene filtering using t-test
3. **Group**: Group genes based on gene-group coupling data
4. **Score**: Evaluate groups using ML models
5. **Model**: Train ML models on best-performing groups
6. **Predict**: Generate predictions using trained models

## Project Structure
```
src/
├── filtering/               # Gene filtering logic
├── grouping/                # Gene grouping implementation
├── scoring/                 # Group scoring algorithms
├── modeling/                # ML model training
├── machine_learning/        # Include machine learning tools
├── feature_selection/       # FS Tools
└── utils/                   # Shared utilities
data/
├── grouping_data/
├── main_data/
├── test/                    # Test data for a fast preliminary check
tests/                       # Python tests
```

## Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning
- matplotlib/seaborn: Visualization
- jupyter: Development environment
- dask: Large dataset handling

## Installation
1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```
2. Install the required dependencies:
    ```sh
    pip install -r dependencies.txt
    ```

## Usage
### Running the Pipeline
1. Load input data:
    ```python
    import pandas as pd
    from config import INPUT_EXPRESSION_DATA, INPUT_GROUP_DATA
    from utils.logger import setup_logger
    from GSM_pipeline import gsm_run, GSMConfig

    expression_data = pd.read_csv(INPUT_EXPRESSION_DATA)
    group_data = pd.read_csv(INPUT_GROUP_DATA)
    logger = setup_logger()

    # Configure and run pipeline
    config = GSMConfig(sample_ratio=0.8, n_iteration_workflow=5)
    gsm_run(expression_data, group_data, logger, config)
    ```

### Example Jupyter Notebook
You can find an example Jupyter notebook in `src/main.ipynb` that demonstrates how to run the GSM pipeline step-by-step.

## Documentation
- Maintain comprehensive docstrings
- Include usage examples
- Reference official library documentation
- Document data structures and workflows
- At the top of each file, include a summary of the file's purpose and functionality along with:
  - File's primary purpose and role in the pipeline, briefly
  - List of key functions/classes with brief descriptions
  - Usage examples where appropriate
  - Any important notes or caveats

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or issues, please open an issue on the repository or contact the project maintainers.
