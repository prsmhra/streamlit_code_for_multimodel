
import os
import json
import re
import csv
import logging
from google import genai
from typing import Dict, List, Optional, Tuple, Literal
from src.python.app.multimodal.multimodal_instruction import MULTIMODAL_VALIDATION_PROMPT,MULTIMODAL_ALIGNMENT_PROMPT
from src.python.app.constants.constants import Constants
logger = logging.getLogger(__name__)



def extract_json_from_text(text: str) -> Optional[Dict]:
    """Robustly extract JSON from text with markdown formatting."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Fallback for simple JSON string
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    logger.warning(f"Failed to extract any JSON from text: {text[:200]}...")
    return None


def safe_upload_file(client: genai.Client, file_path: str, file_type: str = "audio"):
    """Upload file with validation. Uses client.files.upload."""
    logger.info(f"Uploading {file_type}: {file_path}")
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > Constants.FILE_SIZE_LIMIT:
        logger.error(f"File too large: {file_size_mb:.2f}MB (max 100MB)")
        raise ValueError(f"File too large: {file_size_mb:.2f}MB (max 100MB)")
    
    # Use the client.files.upload method as seen in your script
    uploaded = client.files.upload(file=file_path)
    logger.info(f"Uploaded {file_type}: {getattr(uploaded, Constants.NAME_KEY, 'unknown_name')}")
    return uploaded


def validate_medical_relevance_prompt_only(client: genai.Client, user_prompt: str, model_name: str) -> Dict:
    """
    STEP 1: Validate if user prompt is medical-related - WITHOUT ANY MEDIA.
    """

    logger.info("[Validating] STEP 1: Validating medical relevance of prompt...")
    
    validation_prompt=MULTIMODAL_VALIDATION_PROMPT.format(user_prompt=user_prompt)
    
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[validation_prompt]
        )
        
        validation_data = extract_json_from_text(response.text.strip())
        
        if not validation_data:
            logger.error("Failed to parse validation response from model.")
            return {
                Constants.IS_MEDICAL_KEY: False,
                Constants.REASONING_KEY: "Failed to parse validation response",
                Constants.ERROR_KEY: True
            }
        
        return validation_data
        
    except Exception as e:
        logger.error(f"[Failed] Prompt validation error: {str(e)}")
        return {
            Constants.IS_MEDICAL_KEY: False,
            Constants.REASONING_KEY: f"Validation failed: {str(e)}",
            Constants.ERROR_KEY: True
        }

def validate_csv_has_data(csv_path: str) -> Dict:
    """
    Validate if CSV file exists and contains actual data.
    """
    logger.info(f"[Validating] Validating CSV file: {csv_path}")
    
    if not os.path.exists(csv_path):
        return {Constants.IS_VALID_KEY: False, Constants.HAS_DATA_KEY: False, Constants.ERROR_KEY: "file_not_found"}
    
    file_size = os.path.getsize(csv_path)
    if file_size == 0:
        return {Constants.IS_VALID_KEY: False, Constants.HAS_DATA_KEY: False, Constants.ERROR_KEY: "empty_file"}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) <= 1:
                return {Constants.IS_VALID_KEY: False, Constants.HAS_DATA_KEY: False, Constants.ERROR_KEY: "only_header_or_no_rows"}
            
            headers = rows[0]
            non_empty_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]
            
            if len(non_empty_rows) == 0:
                return {Constants.IS_VALID_KEY: False, Constants.HAS_DATA_KEY: False, Constants.ERROR_KEY: "empty_data_rows"}
            
            return {
                Constants.IS_VALID_KEY: True,
                Constants.HAS_DATA_KEY: True,
                Constants.ERROR_KEY: None,
                Constants.DATA_ROW_COUNT_KEY: len(non_empty_rows),
            }
            
    except Exception as e:
        return {Constants.IS_VALID_KEY: False, Constants.HAS_DATA_KEY: False, Constants.ERROR_KEY: f"read_error: {str(e)}"}


def validate_audio_content_alignment(
    client: genai.Client,
    uploaded_audio,
    user_prompt: str,
    disease_focus: str,
    model_name: str
) -> Dict:
    """
    Validate if audio content actually matches what the prompt is asking for.
    """
    
    logger.info("[Validating] Validating audio content alignment...")

    if disease_focus=='':
        alignment_prompt=MULTIMODAL_ALIGNMENT_PROMPT.format(user_prompt=user_prompt, 
        disease_focus="GENERAL MEDICAL ANALYSIS")
    else:
        disease_prompt=f"Disease Name:{disease_focus}"
        alignment_prompt=MULTIMODAL_ALIGNMENT_PROMPT.format(user_prompt=user_prompt, 
        disease_focus=disease_prompt)
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[alignment_prompt, uploaded_audio]
        )
        
        alignment_data = extract_json_from_text(response.text.strip())
        
        if not alignment_data:
            return {
                Constants.IS_ALIGNED_KEY: False,
                Constants.REASONING_KEY: "Failed to parse alignment validation response",
                Constants.ERROR_KEY: True
            }
        
        return alignment_data
        
    except Exception as e:
        logger.error(f"[Failed] Content alignment validation error: {str(e)}")
        return {
            Constants.IS_ALIGNED_KEY: False,
            Constants.REASONING_KEY: f"Alignment validation failed: {str(e)}",
            Constants.ERROR_KEY: True
        }
