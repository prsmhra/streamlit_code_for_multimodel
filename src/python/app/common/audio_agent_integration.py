"""
Audio Agent Integration for Medical AI Agent Pipeline
Integrates the complete audio processing pipeline from app.py into the multi-modal agent system.
Uses existing audio components: cli, orchestrator, extractor, summary, etc.
"""

import os
import sys
import json
import math
import tempfile
import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

# Import audio pipeline components from your existing system
from Config import config
from src.python.app.common.cli import main as cli_main
from src.python.app.common.summary import generate_frame_wise_summaries
from src.python.app.constants.constants import Constants
from Config import config

logger = config.get_logger(__name__)


class AudioAgentPipeline:
    """
    Audio processing pipeline that integrates with the Medical AI Agent system.
    This wraps your existing audio pipeline (app.py) into the agent framework.
    """
    
    def __init__(self):
        """Initialize the audio agent pipeline."""
        self.work_dir = Constants.AUDIO_OUT_DIR
        # os.makedirs(self.work_dir, exist_ok=True)
        self.gemini_client = None
        self.tool_placeholders = []  # For UI updates
        
    def set_ui_placeholders(self, placeholders: List):
        """Set UI placeholder references for real-time updates."""
        self.tool_placeholders = placeholders
        
    def load_audio_once(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file once into memory for efficient batch processing."""
        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            logger.info(f"Audio loaded: duration={len(y)/sr:.2f}s, sr={sr}")
            return y, sr
        except Exception as e:
            logger.exception(f"Failed to load audio: {e}")
            raise

    def write_wav_segment(self, target_path: Path, y: np.ndarray, sr: int) -> None:
        """Write audio segment using soundfile with scipy fallback."""
        try:
            sf.write(str(target_path), y, sr, subtype="PCM_16")
            logger.info(f"Segment written: {target_path}")
        except Exception:
            try:
                from scipy.io import wavfile
                int_data = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
                wavfile.write(str(target_path), sr, int_data)
                logger.info(f"Segment written (scipy): {target_path}")
            except Exception as e:
                logger.error(f"Failed to write WAV: {e}")
                raise RuntimeError("No suitable WAV writer: install soundfile or scipy")

    def process_audio_batches(self, 
                              audio_path: str,
                              y_full: np.ndarray,
                              sr: int,
                              batch_seconds: float = 5.0,
                              overlap_seconds: float = 0.0) -> List[Dict[str, Any]]:
        """
        Generate batch specifications with overlap support.
        Returns list of dicts with batch metadata.
        """
        total_s = float(len(y_full)) / float(sr)
        
        # Validation
        if batch_seconds <= 0:
            raise ValueError("batch_seconds must be > 0")
        if overlap_seconds < 0 or overlap_seconds >= batch_seconds:
            raise ValueError("overlap_seconds must be >= 0 and < batch_seconds")
        
        step = batch_seconds - overlap_seconds
        if step <= 0:
            raise ValueError("Invalid overlap: step must be > 0")
        
        n_batches = int(math.ceil((total_s - overlap_seconds) / step))
        batches = []
        
        for b in range(n_batches):
            start_s = float(b * step)
            dur_s = min(batch_seconds, max(0.0, total_s - start_s))
            
            if dur_s <= 0:
                logger.debug(f"Skipping batch {b}: zero duration")
                continue
            
            start_sample = int(start_s * sr)
            end_sample = int((start_s + dur_s) * sr)
            
            batches.append({
                "batch_num": b + 1,
                "start_s": start_s,
                "duration_s": dur_s,
                "start_sample": start_sample,
                "end_sample": end_sample
            })
        
        logger.info(f"Generated {len(batches)} batches: total={total_s:.2f}s, "
                    f"batch_size={batch_seconds}s, overlap={overlap_seconds}s")
        return batches

    def extract_batch_audio(self, y_full: np.ndarray, batch_spec: Dict[str, Any]) -> np.ndarray:
        """Extract audio slice for a batch from pre-loaded audio."""
        start = batch_spec["start_sample"]
        end = batch_spec["end_sample"]
        return y_full[start:end]

    def get_batch_output_dir(self, batch_num: int, file_name: str) -> str:
        """Create and return output directory for a batch."""
        batch_dir = os.path.join(self.work_dir, f"{file_name.split(Constants.DOT)[Constants.ZERO]}/audio_batch_{batch_num:03d}")
        os.makedirs(batch_dir, exist_ok=True)
        return batch_dir

    async def process_audio_async(self,
                                  audio_path: str,
                                  audio_file_name: str,
                                  batch_seconds: float = 5.0,
                                  overlap_seconds: float = 0.0,
                                  num_features: int = 5,
                                  user_prompt: str = "Analyze this audio data",
                                  progress_callback=None,
                                  status_callback=None) -> Dict[str, Any]:
        """
        Main async method to process audio through the agent pipeline.
        
        Args:
            audio_path: Path to audio file
            batch_seconds: Duration of each batch in seconds
            overlap_seconds: Overlap between batches
            num_features: Number of top features for RF importance
            user_prompt: User's analysis prompt
            progress_callback: Optional callback for progress updates
            status_callback: Optional callback for status updates
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Update status
            if status_callback:
                status_callback("Loading audio...", 1)
            
            # Load audio once
            logger.info(f"Loading audio: {audio_path}")
            y_full, sr = self.load_audio_once(audio_path)
            
            # Update status
            if status_callback:
                status_callback("Calculating batch schedule...", 5)
            
            # Generate batch specifications
            logger.info("Calculating batch schedule...")
            batch_specs = self.process_audio_batches(
                audio_path, y_full, sr,
                batch_seconds=batch_seconds,
                overlap_seconds=overlap_seconds
            )
            
            n_batches = len(batch_specs)
            temp_dir = Path(tempfile.mkdtemp(prefix="audio_batch_"))
            batch_outputs = []
            gemini_texts = []
            
            logger.info(f"Processing {n_batches} audio batches")
            
            # Save user prompt for orchestrator
            promt_dir = f"{self.work_dir}/{audio_file_name.split(Constants.DOT)[Constants.ZERO]}"
            prompt_file = Path(promt_dir) / "prompt.txt"
            os.makedirs(promt_dir, exist_ok=True)
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(user_prompt)
            
            try:
                # Process each batch
                for b_idx, batch_spec in enumerate(batch_specs):
                    batch_num = batch_spec["batch_num"]
                    start_s = batch_spec["start_s"]
                    dur_s = batch_spec["duration_s"]
                    progress = 10 + int((b_idx / n_batches) * 70)
                    
                    if status_callback:
                        status_callback(
                            f"Batch {batch_num}/{n_batches} ({start_s:.1f}s-{start_s+dur_s:.1f}s)",
                            progress
                        )
                    
                    logger.info(f"Processing batch {batch_num}/{n_batches} ({start_s:.1f}s-{start_s+dur_s:.1f}s)")
                    
                    try:
                        # Extract batch audio
                        seg_y = self.extract_batch_audio(y_full, batch_spec)
                        
                        # Write temp segment
                        seg_path = temp_dir / f"audio_batch_{batch_num:03d}_{int(start_s*1000):05d}ms.wav"
                        self.write_wav_segment(seg_path, seg_y, sr)
                        
                        # Process batch with orchestrator
                        batch_out_dir = self.get_batch_output_dir(batch_num, audio_file_name)
                        logger.info(f"Batch Output Dir {batch_out_dir}")
                        logger.info(f"Calling audio orchestrator for batch {batch_num}")
                        
                        # Prepare CLI arguments
                        sys.argv = [
                            "cli_main",
                            "--audio", str(seg_path),
                            "--windows", *map(str, Constants.DEFAULT_WINDOWS),
                            "--top-k", str(num_features),
                            "--out-dir", batch_out_dir
                        ]
                        
                        # Run CLI (synchronously in executor to avoid blocking)
                        await asyncio.get_event_loop().run_in_executor(None, cli_main)
                        
                        # Load result
                        json_path = Path(batch_out_dir) / f"gemini_result_{Path(str(seg_path)).stem}.json"
                        batch_text = "No result"
                        if json_path.exists():
                            with open(json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            batch_text = data.get("gemini_text") or data.get("gemini_text_combined") or "No text"
                            gemini_texts.append(batch_text)
                        
                        batch_outputs.append({
                            "batch_num": batch_num,
                            "start_s": start_s,
                            "duration_s": dur_s,
                            "result": batch_text,
                            "error": None,
                            "batch_dir": batch_out_dir
                        })
                        
                        logger.info(f"Batch {batch_num} completed successfully")
                        
                        if progress_callback:
                            progress_callback(batch_outputs)
                        
                    except Exception as e:
                        logger.exception(f"Batch {batch_num} failed: {e}")
                        batch_outputs.append({
                            "batch_num": batch_num,
                            "start_s": start_s,
                            "duration_s": dur_s,
                            "result": None,
                            "error": str(e),
                            "batch_dir": None
                        })
                        
                        if progress_callback:
                            progress_callback(batch_outputs)
                
                # Generate frame-wise summaries
                if status_callback:
                    status_callback("Generating cross-batch summary...", 85)
                
                logger.info("Generating frame-wise summaries...")
                
                if self.gemini_client is None:
                    self.gemini_client = config.setup_gemini_client()
                
                summary_text = "No summary generated"
                if self.gemini_client:
                    try:
                        summary_result = generate_frame_wise_summaries(
                            out_dir=promt_dir,
                            client=self.gemini_client,
                            model_name=config.DEFAULT_MODEL_NAME
                        )
                        
                        # Extract summary text
                        summary_text = summary_result.get("summary_text", "No summary generated")
                    except Exception as e:
                        logger.error(f"Summary generation failed: {e}")
                        summary_text = f"Summary generation failed: {str(e)}"
                else:
                    summary_text = "Gemini client not available for summary generation"
                
                # Create combined result
                combined_text = None
                if gemini_texts:
                    combined_text = "\n\n--- BATCH BOUNDARY ---\n\n".join(gemini_texts)
                
                combined_result = {
                    "audio_stem": Path(audio_path).stem,
                    "audio_path": str(audio_path),
                    "n_batches": n_batches,
                    "batch_seconds": batch_seconds,
                    "overlap_seconds": overlap_seconds,
                    "batches": batch_outputs,
                    "gemini_text_combined": combined_text,
                    "summary_text": summary_text,
                    "out_dir": promt_dir
                }
                
                # Save combined summary
                combined_path = Path(promt_dir) / f"audio_combined_summary_{Path(audio_path).stem}.json"
                with open(combined_path, "w", encoding="utf-8") as f:
                    json.dump(combined_result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Audio processing complete. Summary saved: {combined_path}")
                
                if status_callback:
                    status_callback("All batches processed!", 100)
                
                return combined_result
                
            finally:
                # Cleanup temp directory
                try:
                    shutil.rmtree(str(temp_dir))
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp: {e}")

                #stop agent loop gracefully
                try:
                    await self._safe_shutdown()
                except:
                    pass
        
        except Exception as e:
            logger.exception(f"Audio processing failed: {e}")
            if status_callback:
                status_callback(f"Error: {str(e)[:50]}", 0)
            raise
    
    async def _safe_shutdown(self):
        """Safely close the event loop after async operations without killing Streamlit."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                current_task = asyncio.current_task(loop=loop)
                for task in asyncio.all_tasks(loop):
                    # Don't cancel the main running task
                    if task is not current_task and not task.done():
                        task.cancel()
                logger.info("✅ Gracefully stopped pending async tasks (non-main).")
        except asyncio.CancelledError:
            # Prevent propagation of cancellation error to Streamlit
            logger.info("🧩 Async shutdown cancelled cleanly (ignored).")
        except Exception as e:
            logger.warning(f"⚠️ Failed to stop async loop safely: {e}")


# Convenience function for synchronous calling
def process_audio_file(audio_path: str,
                       work_dir: str,
                       batch_seconds: float = 5.0,
                       overlap_seconds: float = 0.0,
                       num_features: int = 5,
                       user_prompt: str = "Analyze this audio data") -> Dict[str, Any]:
    """
    Synchronous wrapper for audio processing.
    
    Args:
        audio_path: Path to audio file
        work_dir: Output directory
        batch_seconds: Duration of each batch
        overlap_seconds: Overlap between batches
        num_features: Number of top features
        user_prompt: Analysis prompt
        
    Returns:
        Processing results dictionary
    """
    pipeline = AudioAgentPipeline()
    pipeline.work_dir = work_dir
    
    result = asyncio.run(pipeline.process_audio_async(
        audio_path=audio_path,
        batch_seconds=batch_seconds,
        overlap_seconds=overlap_seconds,
        num_features=num_features,
        user_prompt=user_prompt
    ))
    
    return result