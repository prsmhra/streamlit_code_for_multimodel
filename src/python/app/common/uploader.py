"""
Lightweight helper module for file uploads and JSON previews.

Functions:
- upload_file_if_client(client, filepath): uploads a file via GenAI client if available.
- make_preview_from_json(json_path, n_frames=6): returns the first N frames or a truncated string preview.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_file_if_client(client, filepath: str):
    """
    Attempt to upload a file via a GenAI client (if provided).

    Args:
        client: GenAI client object (expected to have .files.upload)
        filepath (str): path to file on disk

    Returns:
        upload reference object if successful, or None if failed.
    """
    if client is None:
        logger.debug("No client provided; skipping upload for %s", filepath)
        return None

    try:
        upload_ref = client.files.upload(file=str(filepath))
        logger.info("Uploaded file: %s -> %s", filepath, getattr(upload_ref, "name", str(upload_ref)))
        return upload_ref
    except Exception as e:
        logger.warning("Upload failed for %s: %s", filepath, e)
        return None


def make_preview_from_json(json_path: str, n_frames: int = 6):
    """
    Load a small preview subset of a JSON feature file.
    Falls back to truncated raw text if parsing fails.

    Args:
        json_path (str): path to JSON file on disk
        n_frames (int): number of frames to include from start

    Returns:
        list (first N records) or str (truncated text) or None
    """
    p = Path(json_path)
    if not p.exists():
        logger.warning("JSON file not found for preview: %s", json_path)
        return None

    try:
        with p.open("r", encoding="utf-8") as fh:
            arr = json.load(fh)
        if isinstance(arr, list):
            preview = arr[:n_frames]
            logger.debug("Loaded JSON preview for %s (%d frames)", json_path, len(preview))
            return preview
    except Exception as e:
        logger.debug("JSON load failed for preview (%s): %s", json_path, e)

    # Fallback: raw truncated text
    try:
        text = p.read_text(encoding="utf-8")
        truncated = text[:2000]
        logger.debug("Returning truncated text preview for %s (len=%d)", json_path, len(truncated))
        return truncated
    except Exception as e:
        logger.warning("Could not create preview for %s: %s", json_path, e)
        return None

