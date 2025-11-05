"""
Robust top-level orchestrator for:
 - multi-window feature extraction
 - local RandomForest guidance per window
 - optional upload of audio + feature JSONs to a GenAI client
 - building a strict instruction + compact manifest
 - attempting a compact model call first, with a per-window fallback
This version carefully inspects callable signatures and filters the kwargs
to avoid "unexpected keyword argument" runtime errors that arise across SDK versions.
"""
import inspect
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.python.app.constants.constants import Constants

from src.python.app.common.extractor import extract_features_to_df
from src.python.app.common.rf_analysis import fit_rf_and_get_top_features_from_json
from src.python.app.common.uploader import upload_file_if_client, make_preview_from_json
from src.python.app.common.gemini_client import _extract_text_from_genai_response
from src.python.app.instructions.audio_agent_instructions import SAMPLE_PROMPT

logger = logging.getLogger(__name__)


def _filter_kwargs_for_callable(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new dict containing only keys accepted by callable_obj's signature.
    If signature inspection fails, return kwargs unchanged (best-effort).
    """
    try:
        sig = inspect.signature(callable_obj)
        accepted = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return filtered
    except Exception:
        # If we can't inspect the signature, return the original kwargs and let the call possibly fail.
        return kwargs.copy()


def _attempt_call(callable_obj, payload: Dict[str, Any]) -> Tuple[Any, Optional[Exception]]:
    """
    Try to call callable_obj with a payload filtered to parameters the callable accepts.
    Returns (result, None) on success or (None, Exception) on failure.
    """
    try:
        filtered = _filter_kwargs_for_callable(callable_obj, payload)
        res = callable_obj(**filtered)
        return res, None
    except Exception as e:
        return None, e


def _try_call_variants(client: Any, model_name: str, contents: Any, extra_opts: Dict[str, Any]) -> Tuple[Any, str, Optional[Exception]]:
    """
    Try a small suite of plausible callables / parameter shapes on the provided client.
    Returns (response_object, method_label, None) on success or (None, last_method_label, last_exception) on failure.
    """
    last_exc = None
    last_label = "none"

    # Prepare a set of candidate (callable, payload, label) tuples to try.
    # For payload we include model name where appropriate and put the 'contents' into plausible param names.
    candidate_attempts = []

    # Common payload templates (we will always filter these to the callable's signature)
    base_payload = {"model": model_name}
    # Most SDK variants expect one of: contents, input, messages, prompt
    candidate_attempts.append(("client.models.generate_content", getattr(getattr(client, "models", None), "generate_content", None),
                               {**base_payload, "contents": contents, **extra_opts}))
    candidate_attempts.append(("client.models.generate", getattr(getattr(client, "models", None), "generate", None),
                               {**base_payload, "prompt": contents, **extra_opts}))
    candidate_attempts.append(("client.generate", getattr(client, "generate", None),
                               {**base_payload, "contents": contents, **extra_opts}))
    candidate_attempts.append(("client.responses.create", getattr(getattr(client, "responses", None), "create", None),
                               {**base_payload, "input": contents, **extra_opts}))
    candidate_attempts.append(("client.responses.create.messages", getattr(getattr(client, "responses", None), "create", None),
                               {**base_payload, "messages": contents, **extra_opts}))
    candidate_attempts.append(("client.responses.generate", getattr(getattr(client, "responses", None), "generate", None),
                               {**base_payload, "contents": contents, **extra_opts}))
    # Last-resort: try to call the client directly if it is callable
    if callable(client):
        candidate_attempts.append(("client(callable)", client, {**base_payload, "contents": contents, **extra_opts}))

    for label, cand, payload in candidate_attempts:
        if cand is None:
            continue
        last_label = label
        try:
            logger.debug(f"Trying callable: {label} with payload keys: {list(payload.keys())}")
            resp, err = _attempt_call(cand, payload)
            if err is None:
                return resp, label, None
            else:
                last_exc = err
                logger.debug(f"Callable {label} failed: {err}")
                # keep trying
        except Exception as e:
            last_exc = e
            logger.debug(f"Exception while trying {label}: {e}", exc_info=True)
            continue

    return None, last_label, last_exc


def run_all_windows_and_call_gemini(audio_path: str,
                                    windows_to_try: List[float],
                                    hop_ratio: float = Constants.DEFAULT_HOP_RATIO,
                                    out_dir: str = ".",
                                    model_name: str = Constants.MODEL_NAME,
                                    client: Any = None,
                                    call_model_if_available: bool = True,
                                    rf_top_k: int = 10) -> Dict[str, Any]:
    """
    Orchestrator:
      1. Extract features at requested windows and save CSV/JSON
      2. Fit RF per-window to provide top-K features (guidance)
      3. Upload audio + JSONs (if client provided), otherwise produce small previews
      4. Build a strict instruction + compact manifest (prompt read from file if available)
      5. Attempt a compact call (all windows) first, else fallback to per-window calls
    """
    #     # Resolve model name (user-selected or default)
    # if not model_name:
    #     from Config.config import DEFAULT_MODEL_NAME
    #     model_name = DEFAULT_MODEL_NAME
    #     logger.info(f"[model] Using default model: {model_name}")
    # else:
    #     logger.info(f"[model] Using user-selected model: {model_name}")

    audio_name = out_dir.split(os.sep)[-Constants.ONE]
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    per_window: Dict[float, Dict[str, Any]] = {}

    # 1) Feature extraction
    for w in windows_to_try:
        logger.info(f"[extract] running extraction for audio: {audio_path}")
        df, sr = extract_features_to_df(audio_path, frame_seconds=w, hop_ratio=hop_ratio)
        stem = Path(audio_path).stem
        csv_path = out_dir / f"features_{stem}_{int(w*1000)}ms.csv"
        json_path = out_dir / f"features_{stem}_{int(w*1000)}ms.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        per_window[w] = {"csv_path": str(csv_path), "json_path": str(json_path), "n_frames": int(len(df)), "sr": int(sr)}
        logger.info(f"[extract] {int(w*1000)}ms -> frames={per_window[w]['n_frames']}, csv={csv_path}, json={json_path}")

    # 2) RF local guidance
    rf_guidance: Dict[float, List[Dict[str, float]]] = {}
    for w, info in per_window.items():
        try:
            top_pairs = fit_rf_and_get_top_features_from_json(info["json_path"], top_k=rf_top_k)
            rf_guidance[w] = [{"feature": fp, "importance": float(im)} for fp, im in top_pairs]
            logger.info(f"[rf] window {w}s -> top features: {[fp for fp, _ in top_pairs]}")
        except Exception as e:
            logger.warning(f"[rf] failed for window {w}s: {e}")
            rf_guidance[w] = []

    # 3) Upload audio (if client exists)
    audio_upload_ref = None
    if client is not None:
        try:
            logger.info("[upload] uploading audio...")
            audio_upload_ref = upload_file_if_client(client, audio_path)
            logger.info(f"[upload] audio uploaded: {getattr(audio_upload_ref, 'name', audio_upload_ref)}")
        except Exception as e:
            logger.warning(f"[upload] audio upload failed: {e}")
    else:
        logger.info("[upload] no client; skipping upload.")

    # 4) Upload feature JSONs or create previews
    uploaded_feature_files: Dict[float, Dict[str, Any]] = {}
    for w, info in per_window.items():
        json_path = info["json_path"]
        rf_top = rf_guidance.get(w, [])
        if client is not None:
            try:
                up_ref = upload_file_if_client(client, json_path)
                uploaded_feature_files[w] = {"uploaded": True, "ref": up_ref, "meta": info, "rf_top": rf_top}
                continue
            except Exception as e:
                logger.warning(f"[upload] failed to upload {json_path}: {e}")
        # fallback
        preview = make_preview_from_json(json_path)
        uploaded_feature_files[w] = {"uploaded": False, "ref": None, "meta": info, "preview": preview, "rf_top": rf_top}

    # 5) Load user prompt (or fallback to default)
    prompt_file_path = Path(out_dir) / "prompt.txt"
    if prompt_file_path.exists():
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                model_instructions = f.read().strip()
            if not model_instructions:
                logger.warning("[prompt] Empty prompt.txt, using default instructions.")
                model_instructions = SAMPLE_PROMPT
            else:
                logger.info(f"[prompt] Loaded user prompt from {prompt_file_path}")
        except Exception as e:
            logger.warning(f"[prompt] Failed to read prompt.txt: {e}")
            model_instructions = SAMPLE_PROMPT
    else:
        logger.warning("[prompt] No prompt.txt found, using default instructions.")
        model_instructions = SAMPLE_PROMPT
    # Append the audio filename reference for context
    model_instructions += f"\n\nThe audio file: '{Path(audio_path).name}'"

    # 6) Build manifest
    manifest_entries = []
    for w, d in uploaded_feature_files.items():
        meta = d.get("meta", {})
        sr = meta.get("sr")
        n_frames = meta.get("n_frames")
        rf_top = d.get("rf_top", [])
        rf_str = ", ".join([f"{r['feature']}({r['importance']:.4f})" for r in rf_top]) if rf_top else "none"

        if d.get("uploaded"):
            ref_name = getattr(d["ref"], "name", str(d["ref"]))
            manifest_entries.append(f"- window_s={w:.3f}: uploaded_file={ref_name}, frames={n_frames}, sr={sr}, rf_top=[{rf_str}]")
        else:
            preview_len = len(d.get("preview") or [])
            manifest_entries.append(f"- window_s={w:.3f}: preview_len={preview_len}, frames={n_frames}, sr={sr}, rf_top=[{rf_str}]")

    manifest_text = "Feature files manifest:\n" + "\n".join(manifest_entries)

    # 7) Assemble compact content
    contents_compact: List[Any] = [model_instructions, manifest_text]
    if audio_upload_ref is not None:
        contents_compact.append(audio_upload_ref)
    else:
        contents_compact.append(f"AUDIO_FILENAME: {Path(audio_path).name}")

    for w, d in uploaded_feature_files.items():
        if not d.get("uploaded"):
            preview_text = json.dumps(d.get("preview")) if not isinstance(d.get("preview"), str) else d["preview"]
            contents_compact.append(f"PREVIEW window_s={w:.3f}: {preview_text}")
            if d.get("rf_top"):
                contents_compact.append(f"RF_GUIDANCE window_s={w:.3f}: {json.dumps(d.get('rf_top'))}")

    # 8) Call Gemini
    extra_opts = {"temperature": 0.0, "top_k": 40}
    response_obj = None
    if call_model_if_available and client is not None:
        logger.info("[attempt] sending compact manifest call to model...")
        resp, method_used, exc = _try_call_variants(client, model_name, contents_compact, extra_opts)
        if resp is not None:
            response_obj = resp
            logger.info(f"[success] {method_used} succeeded.")
        else:
            logger.warning(f"[fail] compact attempt failed: {exc}")

    # 9) Handle response
    if response_obj is not None:
        try:
            gemini_text = _extract_text_from_genai_response(response_obj) or str(response_obj)
        except Exception:
            gemini_text = str(response_obj)

        out_response_path = out_dir / f"gemini_response_{Path(audio_path).stem}.txt"
        out_response_path.write_text(str(gemini_text), encoding="utf-8")
        logger.info(f"[saved] Gemini response saved to {out_response_path}")

        parsed = None
        try:
            parsed = json.loads(gemini_text)
            logger.info("[parse] JSON parsed successfully.")
        except Exception:
            logger.warning("[parse] Could not parse model output as JSON.")

        result = {
            "per_window": per_window,
            "uploaded_feature_files": {
                k: getattr(v["ref"], "name", str(v["ref"])) if v.get("uploaded") else f"preview_{k}"
                for k, v in uploaded_feature_files.items()
            },
            "gemini_text": gemini_text,
            "gemini_json_parsed": parsed,
        }
        (out_dir / f"gemini_result_{Path(audio_path).stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    # 10) If model not called
    result = {
        "per_window": per_window,
        "uploaded_feature_files": {k: f"local_json:{v['json_path']}" for k, v in per_window.items()},
        "gemini_text": None,
        "note": "Model not called (client missing or call_model_if_available=False)."
    }
    (out_dir / f"gemini_result_{Path(audio_path).stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

