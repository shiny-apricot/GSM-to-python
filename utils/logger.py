"""
Logger class
================

This class provides a simple way to log messages to a file. It uses the
logging module to do the heavy lifting.

"""
import logging
import os

class Logger:
    """
    A simple logger class that logs messages to a file.
    """
    def __init__(self, name):
        """
        Initialize the logger.

        :param name: The name of the logger.
        """
        self.name = name
        self.logger = self._configure_logger()

    def _configure_logger(self):
        """
        Configure the logger.

        This method creates a logger, sets its level to DEBUG, creates a file
        handler, sets the formatter for the handler, and adds the handler to
        the logger.
        """
        # Create a logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        # Create a file handler
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        log_file = os.path.join(logs_dir, "app.log")
        file_handler = logging.FileHandler(log_file)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)

        return logger

    def debug(self, msg):
        """
        Log a message at the DEBUG level.

        :param msg: The message to log.
        """
        self.logger.debug(msg)

    def info(self, msg):
        """
        Log a message at the INFO level.

        :param msg: The message to log.
        """
        self.logger.info(msg)

    def warning(self, msg):
        """
        Log a message at the WARNING level.

        :param msg: The message to log.
        """
        self.logger.warning(msg)

    def error(self, msg):
        """
        Log a message at the ERROR level.

        :param msg: The message to log.
        """
        self.logger.error(msg)

    def critical(self, msg):
        """
        Log a message at the CRITICAL level.

        :param msg: The message to log.
        """
        self.logger.critical(msg)
