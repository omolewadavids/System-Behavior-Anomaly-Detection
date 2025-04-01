import logging
import os
from datetime import datetime

# Setup logger
def setup_logger(log_dir):
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    # Create a file handler to store logs in log file
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    # Create a console handler to display logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Create a logger instance
logger = setup_logger(log_dir="logs")

