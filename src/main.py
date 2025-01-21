from config import INPUT_EXPRESSION_DATA, INPUT_GROUP_DATA
from data_processing import data_loader
import GSM_pipeline
import pathlib
from utils import logger

project_folder = pathlib.Path().resolve().parent
print(f"Project Folder: {project_folder}")

# Now you can import the GSM_pipeline module
input_file = project_folder / INPUT_EXPRESSION_DATA
group_file = project_folder / INPUT_GROUP_DATA

input_data = data_loader.load_input_file(input_file)
group_data = data_loader.load_group_file(group_file)

GSM_pipeline.gsm_run(input_data, group_data, logger=logger)