"""
summary.py - Upload Gemini response text files from all batches and generate a combined summary.

This module:
1. Collects all batch folders (batch_001, batch_002, etc.)
2. Finds gemini_response_batch_XXX_00000ms.txt files in each batch
3. Uploads all text files to Gemini
4. Generates a comprehensive cross-batch summary
5. Returns results in backward-compatible format for Streamlit
"""

import json
import logging
import re
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import inspect
from dotenv import load_dotenv
from google import genai
from src.python.app.constants.constants import Constants

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Client setup
# ------------------------------------------------------------------------------

def setup_gemini_client() -> Optional[Any]:
    """Setup Gemini client with API key from .env file."""
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found in environment variables.")
            return None
        client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to setup Gemini client: {e}")
        return None

# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------

def _filter_kwargs_for_callable(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        return kwargs.copy()

def _attempt_call(callable_obj, payload: Dict[str, Any]) -> Tuple[Any, Optional[Exception]]:
    try:
        filtered = _filter_kwargs_for_callable(callable_obj, payload)
        res = callable_obj(**filtered)
        return res, None
    except Exception as e:
        return None, e

def _try_call_variants(client: Any, model_name: str, contents: Any, extra_opts: Dict[str, Any]) -> Tuple[Any, str, Optional[Exception]]:
    """Try multiple possible Gemini API call formats."""
    last_exc = None
    last_label = "none"
    candidate_attempts = []
    base_payload = {"model": model_name}

    try:
        candidate_attempts.append(("client.models.generate_content",
                                   getattr(client.models, "generate_content", None),
                                   {**base_payload, "contents": contents, **extra_opts}))
    except Exception:
        pass

    candidate_attempts.append(("client.responses.create",
                               getattr(getattr(client, "responses", None), "create", None),
                               {**base_payload, "input": contents, **extra_opts}))

    for label, cand, payload in candidate_attempts:
        if not cand:
            continue
        try:
            resp, err = _attempt_call(cand, payload)
            if err is None:
                return resp, label, None
            last_exc = err
        except Exception as e:
            last_exc = e
            continue

    return None, last_label, last_exc

def _extract_text_from_response(response: Any) -> Optional[str]:
    """Extract plain text from Gemini response."""
    if response is None:
        return None
    try:
        if hasattr(response, "text") and response.text:
            return str(response.text)
    except Exception:
        pass
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = []
            for c in candidates:
                content = getattr(c, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        t = getattr(p, "text", None) or getattr(p, "content", None)
                        if t:
                            parts.append(str(t))
            if parts:
                return "\n\n".join(parts)
    except Exception:
        pass
    try:
        if isinstance(response, dict):
            for key in ("text", "output", "content", "payload"):
                if key in response:
                    return str(response[key])
    except Exception:
        pass
    return str(response)

# ------------------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------------------

def collect_batch_folders(out_dir: str) -> List[Path]:
    """Collect all batch folders (batch_001, batch_002, etc.)"""
    out_path = Path(out_dir)
    if not out_path.exists():
        logger.warning(f"⚠️  Output directory does not exist: {out_dir}")
        return []
    
    all_items = list(out_path.iterdir())
    logger.info(f"   All items in {out_dir}: {[item.name for item in all_items if item.is_dir()]}")
    
    batch_folders = [f for f in all_items if f.is_dir() and re.match(r'audio_batch_\d+', f.name.lower())]
    batch_folders.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))
    
    logger.info(f"📁 Found {len(batch_folders)} batch folders: {[f.name for f in batch_folders]}")
    
    return batch_folders

def collect_response_text_files(batch_folders: List[Path]) -> List[Path]:
    """
    Collect gemini_response_batch_XXX_00000ms.txt files from all batches.
    
    Pattern: gemini_response_batch_001_00000ms.txt
    """
    text_files = []
    
    logger.info(f"\n🔍 Collecting Gemini response text files from {len(batch_folders)} batch folders...")
    
    for batch_folder in batch_folders:
        # Look for files matching pattern: gemini_response_*.txt
        pattern = "gemini_response_*.txt"
        matching_files = list(batch_folder.glob(pattern))
        
        logger.info(f"\n   📂 {batch_folder.name}: Found {len(matching_files)} text file(s)")
        
        if len(matching_files) == 0:
            logger.warning(f"      ⚠️  No gemini_response text files in {batch_folder.name}")
            # List all files for debugging
            all_files = list(batch_folder.glob("*.txt"))
            logger.warning(f"      All .txt files ({len(all_files)}): {[f.name for f in all_files[:10]]}")
            continue
        
        for txt_file in matching_files:
            logger.info(f"      ✓ Found: {txt_file.name}")
            text_files.append(txt_file)
    
    # Sort by batch folder name (chronological order)
    text_files.sort(key=lambda x: x.parent.name)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 COLLECTION SUMMARY:")
    logger.info(f"{'='*60}")
    logger.info(f"🟢 Total text files collected: {len(text_files)}")
    
    for txt_file in text_files:
        logger.info(f"   • {txt_file.name} ({txt_file.parent.name})")
    
    return text_files

# ------------------------------------------------------------------------------
# File upload and summary generation
# ------------------------------------------------------------------------------

def upload_files_to_client(client: Any, files: List[Path]) -> List[Any]:
    """Upload text files to Gemini client."""
    from .uploader import upload_file_if_client
    uploaded_refs = []
    for file_path in files:
        try:
            ref = upload_file_if_client(client, str(file_path))
            uploaded_refs.append(ref)
            logger.info(f"✅ Uploaded: {file_path.name} from {file_path.parent.name}")
        except Exception as e:
            logger.error(f"❌ Failed to upload {file_path}: {e}")
    return uploaded_refs


def generate_cross_batch_summary(text_files: List[Path], client: Any, model_name: str,
                                 output_schema: Optional[Dict[str, Any]] = None, 
                                 custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Generate a comprehensive summary across all batch response files."""
    
    if not text_files:
        logger.warning(f"⚠️  No text files provided for summary generation")
        return {"summary_text": "No text files found", "error": "No text files"}

    batch_names = sorted(list(set([f.parent.name for f in text_files])))
    
    logger.info(f"📊 Generating cross-batch summary using {len(text_files)} files from {len(batch_names)} batches")
    logger.info(f"   Batches: {batch_names}")
    logger.info(f"   Files: {[f'{f.name} ({f.parent.name})' for f in text_files]}")

    file_details = "\n".join([f" - {f.name} (from {f.parent.name})" for f in text_files])

    prompt = custom_prompt or f"""
You are a careful, comprehensive analyst synthesizing acoustic analysis results.

You have been provided {len(text_files)} Gemini response text files from {len(batch_names)} batches ({', '.join(batch_names)}).

Each text file contains analysis results from a separate batch of recorded audio.
Your task is to synthesize insights **across all batches**, creating a unified, comprehensive summary.

Focus on:
1. **Cross-batch patterns and trends**: Identify common themes, recurring observations, and patterns across all batches
2. **Batch-to-batch consistency or variations**: Note where batches agree or differ in their findings
3. **Progressive changes**: Detect any temporal trends or evolution across batches (e.g., gradual changes in acoustic properties)
4. **Key insights and findings**: Highlight the most significant discoveries and patterns
5. **Overall conclusions**: Provide an integrated understanding of the audio corpus as a whole

Do NOT simply list per-batch summaries. Instead, perform integrated cross-batch reasoning to create a cohesive narrative.

Files provided:
{file_details}

Return a well-structured, human-readable summary that synthesizes all the information.
"""

    if output_schema:
        prompt += f"\nFormat response as per schema:\n{json.dumps(output_schema, indent=2)}"

    logger.info(f"🔄 Uploading {len(text_files)} text files...")
    uploaded_refs = upload_files_to_client(client, text_files)
    
    if not uploaded_refs:
        logger.error(f"❌ No files were successfully uploaded")
        return {"summary_text": "Upload failed", "error": "No uploads"}

    logger.info(f"✅ Successfully uploaded {len(uploaded_refs)}/{len(text_files)} files")

    contents = [prompt]
    contents.extend(uploaded_refs)
    
    logger.info(f"🤖 Calling Gemini API for cross-batch summary...")
    response, method_used, exc = _try_call_variants(client, model_name, contents, {"temperature": 0.0, "top_k": 40})

    if response is None:
        logger.error(f"❌ API call failed: {exc}")
        return {"error": str(exc), "files": [str(f) for f in text_files]}

    summary_text = _extract_text_from_response(response)
    try:
        parsed_json = json.loads(summary_text)
    except Exception:
        parsed_json = None

    logger.info(f"✅ Generated cross-batch summary ({len(summary_text)} characters)")

    return {
        "summary_text": summary_text,
        "summary_json_parsed": parsed_json,
        "files_processed": [str(f) for f in text_files],
        "batches": batch_names,
        "num_batches": len(batch_names),
        "num_files_uploaded": len(uploaded_refs),
        "method_used": method_used
    }

# ------------------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------------------

def generate_frame_wise_summaries(out_dir: str, client: Any, model_name: str = "gemini-2.0-flash-exp",
                                  output_schema: Optional[Dict[str, Any]] = None,
                                  custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Main function to generate cross-batch summary from Gemini response text files.
    
    Returns backward-compatible structure for existing Streamlit code.
    """
    logger.info(f"🚀 Starting cross-batch summary generation for: {out_dir}")
    
    batch_folders = collect_batch_folders(out_dir)
    if not batch_folders:
        logger.error("❌ No batch folders found!")
        return {"error": "No batch folders found", "total_batches": 0}
    
    text_files = collect_response_text_files(batch_folders)
    if not text_files:
        logger.error("❌ No Gemini response text files found in any batch folder!")
        return {"error": "No text files found", "total_batches": len(batch_folders)}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing cross-batch summary")
    logger.info(f"{'='*60}")
    
    result = generate_cross_batch_summary(text_files, client, model_name, output_schema, custom_prompt)

    # Save summary to output directory (backward compatible structure)
    out_path = Path(out_dir)
    txt_path = out_path / "cross_batch_summary.txt"
    txt_path.write_text(result.get("summary_text", ""), encoding="utf-8")
    logger.info(f"💾 Saved text summary → {txt_path}")

    # Create backward-compatible return structure
    combined_data = {
        "total_frame_sizes": 1,  # Now we have 1 combined summary instead of per-frame
        "total_batches": len(batch_folders),
        "batch_names": [f.name for f in batch_folders],
        "frame_sizes_ms": ["cross_batch"],  # Placeholder for compatibility
        "frame_results": [result],  # Single result in list for compatibility
        "frames": [result],  # backward compatibility
        # New fields
        "cross_batch_summary": result,
        "summary_text": result.get("summary_text", ""),
    }

    json_path = out_path / "cross_batch_summary.json"
    json_path.write_text(json.dumps(combined_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"💾 Saved JSON summary → {json_path}")

    # Print summary to console
    logger.info(f"\n{'='*60}")
    logger.info(f"📝 CROSS-BATCH SUMMARY")
    logger.info(f"{'='*60}")
    print("\n" + result.get("summary_text", "No summary generated"))
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Summary generation complete!")

    return combined_data

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Generate cross-batch summary from Gemini response text files")
    parser.add_argument("--out-dir", default=Constants.AUDIO_OUT_DIR)
    parser.add_argument("--model", default=Constants.MODEL_NAME)
    parser.add_argument("--prompt", help="Custom prompt for summary generation")
    parser.add_argument("--schema", help="Path to JSON schema file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logger.log(f"info inside cli {args.out_dir}")

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    client = setup_gemini_client()
    if not client:
        logger.error("❌ Failed to create Gemini client. Exiting.")
        return
    
    schema = None
    if args.schema and os.path.exists(args.schema):
        with open(args.schema) as f:
            schema = json.load(f)

    result = generate_frame_wise_summaries(args.out_dir, client, args.model, schema, args.prompt)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()