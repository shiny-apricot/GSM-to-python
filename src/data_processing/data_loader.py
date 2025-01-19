import pandas as pd
import pathlib

def load_input_file(input_file_name):
    """
    Load the input file from the data folder.

    Args:
        project_folder (pathlib.Path): The project folder.
        input_file_name (str): The name of the input file.

    Returns:
        pd.DataFrame: The input file data.
    """
    print(input_file_name)
    data = pd.read_csv(input_file_name)
    return data

def load_group_file(group_file_name):
    """
    Load the group file from the data folder.

    Args:
        project_folder (pathlib.Path): The project folder.
        group_file_name (str): The name of the group file.

    Returns:
        pd.DataFrame: The group file data.
    """
    group = pd.read_csv(group_file_name, sep="\t")
    return group