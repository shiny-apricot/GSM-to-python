import pandas as pd
import pathlib

def load_input_file(project_folder, input_file_name):
    """
    Load the input file from the data folder.

    Args:
        project_folder (pathlib.Path): The project folder.
        input_file_name (str): The name of the input file.

    Returns:
        pd.DataFrame: The input file data.
    """
    data_folder = project_folder / "data"
    data_file = data_folder / input_file_name
    data = pd.read_csv(data_file)
    return data

def load_group_file(project_folder, group_file_name):
    """
    Load the group file from the data folder.

    Args:
        project_folder (pathlib.Path): The project folder.
        group_file_name (str): The name of the group file.

    Returns:
        pd.DataFrame: The group file data.
    """
    data_folder = project_folder / "data"
    group_file = data_folder / "group_file" / group_file_name
    group = pd.read_csv(group_file, sep="\t")
    return group