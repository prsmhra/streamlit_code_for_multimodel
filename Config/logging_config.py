import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging():
    """
    Configures the root logger to stop logging to the console
    and start logging to a date-structured file.
    """
    
    # 1. Define the log file path based on the current date
    now= datetime.now()
    log_dir = Path("logs") / now.strftime("%Y") / now.strftime("%m")
    log_filename = now.strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    log_file = log_dir / log_filename

    # 2. Create the directory structure (e.g., logs/2025/11) if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Define the log message format
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 4. Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set the minimum level to log (e.g., INFO, DEBUG)

    # 5. --- IMPORTANT: Stop logging to terminal ---
    # Remove any existing handlers (like the default StreamHandler)
    root_logger.handlers.clear()

    # 6. --- IMPORTANT: Start logging to file ---
    # Create a file handler to write to our log file
    # mode='a' means append, so logs aren't overwritten each time
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    
    # Add the file handler to the root logger
    root_logger.addHandler(file_handler)

    # 7. (Optional) Also log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        # Log the exception
        root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    root_logger.info("Logging setup complete. Logs will be written to: %s", log_file)