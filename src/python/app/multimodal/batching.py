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
    batch_duration_seconds: int
) -> List[Tuple[str, str, str]]:
    """
    Splits a master CSV and a master audio file into time-synchronized batches.

    Args:
        master_csv_path: Path to the full blendshape/AU CSV file.
        master_audio_path: Path to the full WAV audio file.
        batch_duration_seconds: The length of each batch (e.g., 10 seconds).

    Returns:
        A list of tuples. Each tuple contains:
        (batch_label, batch_csv_path, batch_audio_path)
    """
    
    logger.info(f"Starting batch creation for {batch_duration_seconds}s batches.")
    
    # --- 1. Load Master Files ---
    try:
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

    # --- 2. Define Batch Parameters ---
    frames_per_batch = int(Constants.DEFAULT_FPS * batch_duration_seconds)
    batch_duration_ms = int(batch_duration_seconds * Constants.CONSTANT_MS)
    
    total_frames = len(df)
    total_ms = len(audio)
    
    # Calculate total batches based on the *longer* of the two
    num_batches = math.ceil(max(total_frames / frames_per_batch, total_ms / batch_duration_ms))
    
    processed_batches = []
    
    if num_batches == 0:
        logger.warning("No data to batch.")
        return []

    logger.info(f"Calculated {num_batches} batches to create.")

    # --- 3. Loop and Create Batches ---
    for i in range(num_batches):
        start_frame = i * frames_per_batch
        end_frame = (i + 1) * frames_per_batch
        
        start_ms = i * batch_duration_ms
        end_ms = (i + 1) * batch_duration_ms
        
        # --- Create Batch Label and Dir ---
        batch_label = f"batch_{i:03d}_{start_ms // Constants.CONSTANT_MS}s_to_{end_ms // Constants.CONSTANT_MS}s"
        batch_dir = Path(Constants.BATCH_INPUTS_DIR) / batch_label
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 4. Process and Save CSV Batch ---
        batch_csv_path = str(batch_dir / Constants.VISION_BATCH_CSV)
        batch_df = df.iloc[start_frame:end_frame].copy()
        
        # Reset frame_no to be relative to the batch
        if not batch_df.empty and Constants.FRAME_NO_KEY in batch_df.columns:
            batch_df[Constants.FRAME_NO_KEY] = range(1, len(batch_df) + 1)
        
        batch_df.to_csv(batch_csv_path, index=False)
        
        # --- 5. Process and Save Audio Batch ---
        batch_audio_path = str(batch_dir / Constants.AUDIO_WAV_FILE)
        batch_audio = audio[start_ms:end_ms]
        batch_audio.export(batch_audio_path, format=Constants.WAV)
        
        logger.info(f"Created batch: {batch_label} ({len(batch_df)} frames, {len(batch_audio)/1000.0:.2f}s)")
        processed_batches.append((batch_label, batch_csv_path, batch_audio_path))
        
    logger.info(f"Successfully created {len(processed_batches)} batches in {Constants.BATCH_INPUTS_DIR}")
    return processed_batches


