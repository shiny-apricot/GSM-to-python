# G-S-M Bioinformatics Data Pipeline Project
# Purpose: Implementation of Grouping-Scoring-Modeling (GSM) pipeline for gene analysis

## Project Overview
This project implements a modular data pipeline for bioinformatics analysis using the GSM approach:
1. Load: Load and validate input data
2. Filter: Preliminary gene filtering using t-test
3. Group: Group genes based on gene-group coupling data
4. Score: Evaluate groups using ML models
5. Model: Train ML models on best-performing groups
6. Predict: Generate predictions using trained models

## Code Organization Principles

### Architecture
- Modular, file-based organization with clear separation of concerns
- Function-first approach: prefer functions over classes
- Use dataclasses for complex data structures
- NEVER USE Dictionary or Tuple. Instead use dataclass. This makes code more understandable.
- For methods taking more than 2 input parameter, create a config class for the inputs. Name the class as ...Parameters
- Implementation order: skeleton → interfaces → concrete implementations. So do not write everything in one module or function, instead separate it into sub-modules.
- Be simple and understandable so that it can be used and maintained by non-experts.
    - Since my team is not familiar with Python, consider even the people who dont know Python very well.

### Method Organization
- Keep methods focused and concise
- Extract complex logic into separate methods
- Limit method nesting to 2-3 levels
- Break down long methods into smaller, well-named functions
- Use meaningful method names that describe their purpose
- Follow the Single Responsibility Principle
- Always take logger as parameters 

### Code Style
- Follow PEP 8 guidelines
- Use descriptive names for variables, functions, and classes
- Document all files with class/method summaries at the top
- Include input/output examples in docstrings
- When you give input parameters, always to use their names.

### Best Practices
- KISS principle: Keep implementations simple and focused
- Functional programming where appropriate
- Comprehensive error handling and logging
- Unit tests for core functionality

### Performance Guidelines
- Use vectorized operations (numpy/pandas)
- Implement dask for large datasets
- Profile code for optimization
- Minimize memory usage with generators/iterators

## Project Structure
src/
├── filtering/ # Gene filtering logic
├── grouping/ # Gene grouping implementation
├── scoring/ # Group scoring algorithms
├── modeling/ # ML model training
├── machine_learning/ # Include machine learning tools
├── feature_selection # FS Tools
└── utils/ # Shared utilities
data/
├── grouping_data
├── main_data
├── test # Test data for a fast preliminary check
tests/ # Python tests


## Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning
- matplotlib/seaborn: Visualization
- jupyter: Development environment
- dask: Large dataset handling

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
- Do not hesitate to use emojis or ##### type of separators in comments or logs.
- Write documentations for NON-PYTHON researchers