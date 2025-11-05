import logging
import sys
from datetime import datetime
 
def setup_logger(log_file_path=None):
    # Default log file path
    if log_file_path is None:
        log_file_path = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
 
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # capture all levels
 
    # File handler (for all logs)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
 
    # Stream handler (for only final output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)  # only show important info to terminal
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
 
    # Clear any previous handlers and add new ones
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
 
    return logger
 
 