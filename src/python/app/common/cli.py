"""
audio_gemini CLI
- loads GEMINI API key from .env
- tries Colab upload widget
- falls back to asking for a local .wav path
- creates a GenAI client if API key is provided
- runs the orchestrator and saves a sanitized JSON result to disk
"""

from __future__ import annotations

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

from src.python.app.common.orchestrator import run_all_windows_and_call_gemini
from src.python.app.constants.constants import Constants
from Config import config

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
# GEMINI_API_KEY = None
# if GEMINI_API_KEY is None:
    # GEMINI_API_KEY = config.setup_gemini_client()

# --------------------------
# Logging setup
# --------------------------
def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
logger = logging.getLogger("audio_gemini")


# --------------------------
# Utility: Create GenAI client
# --------------------------
def create_client_from_env(api_key: Optional[str] = None):
    try:
        from google import genai  # type: ignore
        if api_key:
            client = genai.Client(api_key=api_key)  # pass directly
            print("[DEBUG] GenAI client created:", client)
            return client
        else:
            print("[DEBUG] No API key provided to create client")
            return None
    except Exception as e:
        print("[DEBUG] Failed to create GenAI client:", e)
        return None



# --------------------------
# Input helpers
# --------------------------
def try_colab_upload() -> Optional[Path]:
    try:
        from google.colab import files  # type: ignore
        logger.info("Please upload a .wav file via the file picker dialog...")
        uploaded = files.upload()
        if uploaded:
            filename = next(iter(uploaded.keys()))
            p = Path(filename).resolve()
            logger.info(f"Uploaded: {p}")
            return p
        return None
    except Exception:
        return None


def prompt_for_audio_path() -> Optional[Path]:
    fallback = input("Enter path to .wav file (or press Enter to abort): ").strip()
    if not fallback:
        logger.info("No audio path entered by user.")
        return None
    p = Path(fallback).expanduser().resolve()
    if not p.exists():
        logger.warning(f"Path does not exist: {p}")
        return None
    return p


# --------------------------
# JSON Serialization Helper
# --------------------------
def _make_json_serializable(obj: Any, _seen: Optional[set] = None) -> Any:
    import numbers
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return f"<circular:{type(obj).__name__}>"
    _seen.add(oid)

    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, numbers.Number):
        try:
            return float(obj)
        except Exception:
            return obj

    if _np is not None:
        try:
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v, _seen) for v in obj]

    try:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            return {k: _make_json_serializable(v, _seen) for k, v in d.items() if not k.startswith("_")}
    except Exception:
        pass

    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


# --------------------------
# Argument parser
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run audio->features->Gemini pipeline (interactive)")
    p.add_argument("--audio", required=False, help="Path to WAV audio file (optional).")
    p.add_argument("--windows", nargs="+", type=float, default = Constants.DEFAULT_WINDOWS)
    # p.add_argument("--audio-file-name", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5, help="Number of top features for RF importance (default: 10)")
    p.add_argument("--out-dir", default=Constants.AUDIO_OUT_DIR)
    p.add_argument("--no-model", action="store_true", help="Do not attempt to create/call GenAI client even if API key provided")
    return p.parse_args()


# --------------------------
# Main function
# --------------------------
def main():
    args = parse_args()
    setup_logging()
    logger.info("Starting audio_gemini pipeline")

    # GEMINI API key
    if GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        logger.info("GEMINI_API_KEY loaded from .env successfully.")
    else:
        logger.warning("No GEMINI_API_KEY found in .env: model calls will be skipped.")

    # Audio input
    audio_path: Optional[Path] = None
    if args.audio:
        p = Path(args.audio).expanduser().resolve()
        if p.exists():
            audio_path = p
            
        else:
            logger.warning(f"Provided audio path does not exist: {p}")

    if audio_path is None:
        audio_path = try_colab_upload() or prompt_for_audio_path()
        if audio_path is None:
            logger.error("No audio file provided. Aborting.")
            sys.exit(1)
            
    # Create GenAI client
    client = None
    if not args.no_model and GEMINI_API_KEY:
        try:
            client = config.setup_gemini_client()
            if client:
                logger.info("[genai] client created successfully.")
            else:
                logger.warning("[genai] client not available in this environment.")
        except Exception as e:
            client = None
            logger.error(f"[genai] failed to create client: {e}")

    # Run orchestrator
    logger.info(f"[extract] running extraction for audio: {audio_path}")
    print(args.top_k)
    result = run_all_windows_and_call_gemini(
        str(audio_path),
        windows_to_try=args.windows,
        rf_top_k=args.top_k,
        out_dir=args.out_dir,
        client=client,
        call_model_if_available=bool(client) and not args.no_model
    )

    # Save JSON result
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_dir) / f"gemini_result_{Path(audio_path).stem}.json"
    try:
        sanitized = _make_json_serializable(result)
        with out_json.open("w", encoding="utf-8") as fh:
            json.dump(sanitized, fh, indent=2, ensure_ascii=False)
        logger.info(f"Saved pipeline result to {out_json}")
    except Exception as e:
        logger.error(f"Failed to write result JSON: {e}")
        fallback_path = Path(args.out_dir) / f"gemini_result_{Path(audio_path).stem}.txt"
        with fallback_path.open("w", encoding="utf-8") as fh:
            fh.write(str(result))
        logger.info(f"Wrote fallback string result to {fallback_path}")

    # Print final output only on terminal
    gem_text = None
    if isinstance(result, dict):
        for key in ("gemini_text", "gemini_text_combined"):
            if key in result:
                gem_text = result[key]
                break

    if gem_text:
        print("\n--- GEMINI RAW OUTPUT (preview) ---\n")
        lines = str(gem_text).splitlines()
        for ln in lines[:800]:
            print(ln)
        if len(lines) > 800:
            print("... (truncated display; full response saved to disk)")
    else:
        print("\nNo model output (offline run or model call skipped). See saved JSON results in", args.out_dir)


# --------------------------
# CLI entry point
# --------------------------
if __name__ == "__main__":
    main()
