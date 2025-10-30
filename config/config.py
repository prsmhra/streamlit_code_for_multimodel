# your_project_root/config/config.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API and Model Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Set it for google-genai library

MODEL_NAME = "gemini-2.0-flash-lite" # Updated to a common model

# --- Application Configuration ---
APP_NAME = "medical_analysis"
USER_ID = "streamlit_user"
INPUT_FPS = 30
DEFAULT_WORK_DIR = "./batch_output"
DEFAULT_BATCH_SIZE = 100

# --- Logging Configuration ---
LOGGING_LEVEL = logging.INFO # Change to logging.DEBUG for more details
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure basic logging ONCE when this module is imported
logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)

# --- Function to get named loggers ---
def get_logger(name):
  """Gets a logger instance with the specified name."""
  return logging.getLogger(name)

# --- Logger for the config module itself (optional) ---
_config_logger = get_logger(__name__) # Use underscore to avoid name clash if needed
_config_logger.info("Configuration loaded.")
# You could add more complex config loading here if needed (e.g., from YAML)

# Optional: Log if API key was loaded successfully
if GOOGLE_API_KEY:
    _config_logger.info("GOOGLE_API_KEY loaded successfully.")