import pandas as pd
from loader import load_input_file, load_group_file
from config import LABEL_OF_NEGATIVE_CLASS, LABEL_OF_POSITIVE_CLASS
from normalization import normalize_data
from train_test_split import train_test_split


class DataPreprocessor:
    def __init__(self, 
                 project_folder, 
                 input_file_name, 
                 group_file_name):
        """
        Initialize the DataPreprocessor.

        Args:
            project_folder (pathlib.Path): The project folder.
            input_file_name (str): The name of the input file.
            group_file_name (str): The name of the group file.
        """
        self.project_folder = project_folder
        self.input_file_name = input_file_name
        self.group_file_name = group_file_name
        self.input_data = load_input_file(project_folder, input_file_name)
        self.group_data = load_group_file(project_folder, group_file_name)

    def convert_labels_to_binary(self,
                                  data,
                                  label_of_negative_class=LABEL_OF_NEGATIVE_CLASS, 
                                  label_of_positive_class=LABEL_OF_POSITIVE_CLASS):
        """
        Convert the labels into binary values.
        """
        data['label'] = data['label'].map({label_of_negative_class: 0, label_of_positive_class: 1})
        return data
    
    def normalize_data(self, 
                       data):
        """
        Normalize the data.

        Returns:
            pd.DataFrame: The normalized data.
        """
        return normalize_data(data)
    
    def train_test_split(self, 
                         data):
        """
        Split the data into training and testing sets.

        Returns:
            pd.DataFrame: The training and testing sets.
        """
        return train_test_split(data)