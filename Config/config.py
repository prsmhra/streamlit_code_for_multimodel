
# your_project_root/config/config.py
import os
import logging
from dotenv import load_dotenv
from Config.logging_config import setup_logging 


# to logs all the info in log file
setup_logging()

# Load environment variables from .env file
load_dotenv()

# --- API and Model Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # We can log this critical error before raising
    logging.critical("GOOGLE_API_KEY environment variable not set.")
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Set it for google-genai library

MODEL_NAME = "gemini-2.0-flash-lite"  # Updated to a common model

# --- Application Configuration ---
APP_NAME = "medical_analysis"
USER_ID = "streamlit_user"


# --- Function to get named loggers ---
def get_logger(name):
    """Gets a logger instance with the specified name."""
    # This will now get a logger that inherits its settings
    # from the root logger configured by setup_logging()
    return logging.getLogger(name)

# --- Logger for the config module itself ---
_config_logger = get_logger(__name__)
_config_logger.info("Configuration loaded.")

# Optional: Log if API key was loaded successfully
if GOOGLE_API_KEY:
    _config_logger.info("GOOGLE_API_KEY loaded successfully.")