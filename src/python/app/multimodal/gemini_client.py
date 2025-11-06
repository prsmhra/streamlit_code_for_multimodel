
import logging
import os
from google import genai
from typing import Any, List, Optional
from src.python.app.constants.constants import Constants
logger = logging.getLogger(__name__)

def create_gemini_client() -> Optional[genai.Client]:
    """
    Creates the centralized genai.Client based on your requested format.
    Reads API key from environment variables.
    """
    api_key = os.getenv(Constants.GOOGLE_API_KEY)
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        logger.info("genai.Client created successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to create genai.Client: {e}")
        return None

async def call_gemini_multimodal(
    client: genai.Client,
    model_name: str,
    prompt_parts: List[Any],
) -> Optional[str]:
    """
    Calls the Gemini model using the client.models.generate_content method.
    """
    logger.info(f"Calling Gemini ({model_name}) with {len(prompt_parts)} parts...")
    
    try:
        # Use the client.models.generate_content format
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_parts
        )
        
        if not response.text:
            logger.warning("Gemini response was empty or blocked.")
            return None
        
        result_text = response.text
        logger.info("Gemini call successful.")
        return result_text
    
    except Exception as e:
        logger.error(f"Gemini API call (client.models.generate_content) failed: {e}")
        return None

def upload_file_with_client(
    client: genai.Client, 
    file_path: str, 
    display_name: str
) -> Optional[Any]: 
    """
    Uploads a file using the client.files.upload method.
    This is a helper for the vision CSV.
    Returns the uploaded file object.
    """
    try:
        logger.info(f"Uploading file: {display_name} from {file_path}")
        # Use the client.files.upload method
        file_obj = client.files.upload(
            file=file_path
        )
        logger.info(f"File uploaded successfully: {file_obj.name}")
        return file_obj
    
    except Exception as e:
        logger.error(f"File upload failed for {file_path}: {e}")
        return None