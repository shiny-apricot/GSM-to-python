"""
logger.py

This module provides a simple logging utility for the GSM bioinformatics data pipeline.
It allows logging messages at different severity levels (info, warning, error) to the console
and to a log file.

Key Functions:
- setup_logger: Configures the logger with specified log level and file output.
- log_info: Logs an informational message.
- log_warning: Logs a warning message.
- log_error: Logs an error message.

Usage Example:
    from utils.logger import setup_logger, log_info, log_warning, log_error

    logger = setup_logger('my_log_file.log')
    log_info("This is an info message.")
    log_warning("This is a warning message.")
    log_error("This is an error message.")
"""

import logging
import sys
from config import LOGGING_LEVEL, LOGGING_FORMAT, LOGGING_OUTPUT_FILE
from IPython import get_ipython


def setup_logger(log_file: str='log.txt', level: int = logging.INFO) -> logging.Logger:
    """Sets up the logger to log messages to both console and a file."""
    logger = logging.getLogger(__name__)
    
    # Prevent duplicate handlers in Jupyter
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add Jupyter output handler using sys.stdout
    jupyter_handler = logging.StreamHandler(sys.stdout)
    jupyter_handler.setFormatter(formatter)
    logger.addHandler(jupyter_handler)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_info(message: str):
    """Logs an informational message."""
    logger = logging.getLogger(__name__)
    logger.info(message)

def log_warning(message: str):
    """Logs a warning message."""
    logger = logging.getLogger(__name__)
    logger.warning(message)

def log_error(message: str):
    """Logs an error message."""
    logger = logging.getLogger(__name__)
    logger.error(message)

def setup_logging():
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, filename=LOGGING_OUTPUT_FILE)
