import streamlit as st
from src.python.app.constants.constants import Constants
from src.python.app.common.web_ui import webUI

st.set_page_config(
    page_title=Constants.APP_NAME,
    page_icon="🎬",
    layout=Constants.WIDE_STR
)
class Inference:
    """
    Entry point class for the Medical AI Agent application.
    """
    def __init__(self):
        """
        Initialize the inference class.
        
        """
        self.webui = webUI()

    def infer(self):
        """
        Start the Streamlit web UI application.
        This method initializes the UI and handles content updates.
        """
        # Render the initial UI content
        self.webui.ui_content()
        
        # Handle any UI updates (e.g., when process button is clicked)
        self.webui.ui_content_updates()