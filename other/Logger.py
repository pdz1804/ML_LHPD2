"""
Logger.py

A class used for logging messages => very useful for debugging

Author: Nguyen Quang Phu
Date: 2025-02-04
"""

import os
import logging

class MyLogger:
    """
    Custom logger class that logs messages to both a file and the console.
    """
    def __init__(self, log_file='app.log'):
        """
        Initialize the logger with a log file.
        Parameters:
        - log_file (str): Path to the log file. Default is 'app.log'.
        """
        self.log_file = log_file
        self._initialize_logger()

    def _initialize_logger(self):
        """
        Set up the logger by creating file and console handlers.
        If the log file already exists, logs are appended to it.
        """
        # Set file mode based on whether the log file already exists
        if os.path.exists(self.log_file):
            file_mode = 'a'
        else:
            file_mode = 'w'

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Set up file handler
        file_handler = logging.FileHandler(self.log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Define log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Clear existing handlers to prevent duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_message(self, message):
        """Log an informational message."""
        self.logger.info(message)

    def change_log_file(self, new_log_file):
        """
        Change the log file and reinitialize the logger.
        Parameters:
        - new_log_file (str): Path to the new log file.
        """
        self.log_file = new_log_file
        self._initialize_logger()

# # Usage
# logger = MyLogger()

# # Enable/Disable tokenizers parallelism to avoid the warning
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
