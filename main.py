"""
Main entry point for the Medical AI Agent Application.
This initializes the web UI and starts the Streamlit application.
"""

import warnings
warnings.filterwarnings("ignore")

from src.python.app.common.inference import Inference


if __name__ == "__main__":
    """
        
    """
    Inference().infer()