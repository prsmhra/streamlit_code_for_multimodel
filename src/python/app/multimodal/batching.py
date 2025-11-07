import logging
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import PydubException

from src.python.app.constants.constants import Constants

logger = logging.getLogger(__name__)

def create_multimodal_batches(
    master_csv_path: str,
    master_audio_path: str,
    batch_duration_seconds: int,
    overlap_seconds: int  
) -> List[Tuple[str, str, str]]:
    """
    Splits a master CSV and a master audio file into time-synchronized batches
    with a specified overlap.

    Args:
        master_csv_path: Path to the full blendshape/AU CSV file.
        master_audio_path: Path to the full WAV audio file.
        batch_duration_seconds: The length of each batch (e.g., 5 seconds).
        overlap_seconds: The amount of overlap (e.g., 1 second).

    Returns:
        A list of tuples. Each tuple contains:
        (batch_label, batch_csv_path, batch_audio_path)
    """
    
    logger.info(f"Starting batch creation: {batch_duration_seconds}s duration, {overlap_seconds}s overlap.")

    # --- 1. Load Master Files ---
    try:
        file_ext = Path(master_csv_path).suffix
        if file_ext == '.xlsx':
            df = pd.read_excel(master_csv_path)
        else:
            df = pd.read_csv(master_csv_path)
        logger.info(f"Loaded master CSV with {len(df)} frames.")
    except Exception as e:
        logger.error(f"Failed to read master CSV: {e}")
        return []

    try:
        audio = AudioSegment.from_file(master_audio_path)
        logger.info(f"Loaded master audio: {len(audio) / 1000.0:.2f}s long.")
    except PydubException as e:
        logger.error(f"Failed to read master audio. Is ffmpeg installed? Error: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to read master audio: {e}")
        return []

    # --- 2. Define Batch and Step Parameters ---
    
    # Validate overlap
    if overlap_seconds >= batch_duration_seconds:
        logger.error(f"Overlap ({overlap_seconds}s) must be less than batch duration ({batch_duration_seconds}s).")
        return []

    # Calculate durations (length of each batch)
    batch_duration_ms = int(batch_duration_seconds * Constants.CONSTANT_MS)
    frames_per_batch = int(Constants.DEFAULT_FPS * batch_duration_seconds)
    
    # Calculate steps (how far to move forward for each new batch)
    step_seconds = batch_duration_seconds - overlap_seconds
    # Ensure step is at least 1ms / 1 frame to avoid division by zero or invalid logic
    step_ms = max(1, int(step_seconds * Constants.CONSTANT_MS)) 
    step_frames = max(1, int(step_seconds * Constants.DEFAULT_FPS))

    total_frames = len(df)
    total_ms = len(audio)
    
    # Calculate total batches based on the *longer* of the two, using the step
    num_batches_audio = math.ceil(total_ms / step_ms)
    num_batches_frames = math.ceil(total_frames / step_frames)
    num_batches = max(num_batches_audio, num_batches_frames)
    
    processed_batches = []
    
    if num_batches == 0:
        logger.warning("No data to batch.")
        return []

    logger.info(f"Calculated {num_batches} batches to create.")

    # --- 3. Loop and Create Batches ---
    for i in range(num_batches):
        # --- Calculate start and end points based on STEP and DURATION ---
        start_frame = i * step_frames
        end_frame = start_frame + frames_per_batch  # Start + Duration
        
        start_ms = i * step_ms
        end_ms = start_ms + batch_duration_ms  # Start + Duration
        
        # --- Create Batch Label and Dir ---
        batch_label = f"batch_{i:03d}_{start_ms // Constants.CONSTANT_MS}s_to_{end_ms // Constants.CONSTANT_MS}s"
        batch_dir = Path(Constants.BATCH_INPUTS_DIR) / batch_label
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 4. Process and Save CSV Batch ---
        # iloc will automatically handle the last partial batch
        batch_df = df.iloc[start_frame:end_frame].copy()
        
        if batch_df.empty and i > 0: # Stop if we're past the end of the CSV
             logger.warning(f"Skipping batch {batch_label}, no frame data found.")
             continue
        
        batch_csv_path = str(batch_dir / Constants.VISION_BATCH_CSV)
        
        # Reset frame_no to be relative to the batch
        if not batch_df.empty and Constants.FRAME_NO_KEY in batch_df.columns:
            batch_df[Constants.FRAME_NO_KEY] = range(1, len(batch_df) + 1)
        
        batch_df.to_csv(batch_csv_path, index=False)
        
        # --- 5. Process and Save Audio Batch ---
        # Audio slicing will automatically handle the last partial batch
        batch_audio = audio[start_ms:end_ms]

        if len(batch_audio) == 0 and i > 0: # Stop if we're past the end of the audio
            logger.warning(f"Skipping batch {batch_label}, no audio data found.")
            continue
            
        batch_audio_path = str(batch_dir / Constants.AUDIO_WAV_FILE)
        batch_audio.export(batch_audio_path, format=Constants.WAV)
        
        logger.info(f"Created batch: {batch_label} ({len(batch_df)} frames, {len(batch_audio)/1000.0:.2f}s)")
        processed_batches.append((batch_label, batch_csv_path, batch_audio_path))
        
    logger.info(f"Successfully created {len(processed_batches)} batches in {Constants.BATCH_INPUTS_DIR}")
    return processed_batches