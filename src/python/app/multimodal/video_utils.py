
import os
import logging
from pathlib import Path
from pydub import AudioSegment
from src.python.app.constants.constants import Constants


logger = logging.getLogger(__name__)

async def extract_audio_from_video(video_path: Path, output_dir: Path) -> str:
    """
    Extracts the audio from a video file and saves it as an MP3 file.

    Args:
        video_path: The Path object to the input video file.
        output_dir: The directory to save the extracted audio file.

    Returns:
        The string path to the newly created MP3 file.
    """
    try:
        logger.info(f"Extracting audio from {video_path.name}...")

        os.makedirs(output_dir, exist_ok=True)
        
        # Define the output path (saving as .mp3)
        output_audio_path = f"{output_dir}{os.sep}{video_path.stem}{Constants.EXTRACTED_AUDIO_SUFFIX}"
        
        # Load the video file (pydub can read audio from video formats)
        audio = AudioSegment.from_file(video_path)
        
        # Export the audio as an MP3 file
        audio.export(output_audio_path, format=Constants.MP3)
        
        logger.info(f"Audio extracted successfully: {output_audio_path}")
        return str(output_audio_path)
        
    except Exception as e:
        logger.error(f"Failed to extract audio from video: {e}")
        raise
