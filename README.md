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

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/shiny-apricot/GSM-to-python.git
    cd GSM-to-python
    ```
2. Install the required dependencies:
    ```sh
    pip install -r dependencies.txt
    ```

## Usage

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
1. Create a new branch (`git checkout -b feature-branch`)
2. Make your changes
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or issues, please open an issue on the repository or contact the project maintainers.
