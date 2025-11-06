"""
Audio pipeline orchestrator.
MODIFIED for multimodal pipeline:
 - Does NOT call Gemini.
 - `run_audio_pipeline_for_batch` returns the 'contents' list
   and file metadata, ready for a top-level orchestrator.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.python.app.multimodal_audio.extractor import extract_features_to_df
from src.python.app.multimodal_audio.rf_analysis import fit_rf_and_get_top_features_from_json
from src.python.app.multimodal_audio.uploader import upload_file_if_client, make_preview_from_json

from src.python.app.constants.constants import Constants  # [Completed] unified import

logger = logging.getLogger(__name__)


def run_audio_pipeline_for_batch(
    audio_path: str,
    windows_to_try: List[float],
    out_dir: str,
    batch_label: str,  # e.g., "0_to_10s"
    client: Any = None,
    hop_ratio: float = Constants.DEFAULT_HOP_RATIO,
    rf_top_k: int = Constants.RF_TOP_K,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Runs audio extraction and preparation for a single batch.
    """
    logger.info(f"[AudioBatch: {batch_label}] Starting processing for {audio_path}")
    os.makedirs(out_dir, exist_ok=True)
    per_window: Dict[float, Dict[str, Any]] = {}

    # 1) Feature extraction
    for w in windows_to_try:
        df, sr = extract_features_to_df(audio_path, frame_seconds=w, hop_ratio=hop_ratio)

        csv_path = f"{out_dir}/features_{batch_label}_{int(w*1000)}ms{Constants.CSV_EXTENSION}"
        json_path = f"{out_dir}/features_{batch_label}_{int(w*1000)}ms{Constants.JSON_EXT}"

        df.to_csv(csv_path, index=False)
        df.to_json(
            json_path,
            orient="records",
            force_ascii=Constants.FORCE_ASCII,
            indent=Constants.JSON_INDENT,
        )

        per_window[w] = {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "n_frames": int(len(df)),
            "sr": int(sr),
        }

        logger.info(
            f"[AudioBatch: {batch_label}] {int(w*1000)}ms -> frames={len(df)}, csv={csv_path}"
        )

    # 2) RF local guidance
    rf_guidance: Dict[float, List[Dict[str, float]]] = {}
    rf_guidance_text_parts = []

    logger.info(f"[AudioBatch: {batch_label}] Generating RF guidance...")
    for w, info in per_window.items():
        try:
            top_pairs = fit_rf_and_get_top_features_from_json(
                info["json_path"], top_k=rf_top_k
            )
            rf_guidance[w] = [
                {"feature": fp, "importance": float(im)} for fp, im in top_pairs
            ]

            rf_text = Constants.RF_GUIDANCE_TEXT_TEMPLATE.format(
                w=w, rf_json=json.dumps(rf_guidance[w])
            )
            rf_guidance_text_parts.append(rf_text)
        except Exception as e:
            logger.warning(f"[AudioBatch: {batch_label}] RF failed for window {w}s: {e}")
            rf_guidance[w] = []

    # 3) Upload raw audio batch
    audio_upload_ref = None
    if client is not None:
        try:
            audio_upload_ref = upload_file_if_client(client, audio_path)
            logger.info(
                f"[AudioBatch: {batch_label}] Audio chunk uploaded: {getattr(audio_upload_ref, 'name', audio_upload_ref)}"
            )
        except Exception as e:
            logger.warning(
                f"[AudioBatch: {batch_label}] Audio chunk upload failed: {e}"
            )
    else:
        logger.info(f"[AudioBatch: {batch_label}] No client; skipping upload.")

    # 4) Upload feature JSONs or create previews
    uploaded_feature_files: Dict[float, Dict[str, Any]] = {}
    for w, info in per_window.items():
        csv_path = info["csv_path"]
        rf_top = rf_guidance.get(w, [])
        if client is not None:
            try:
                up_ref = upload_file_if_client(client, csv_path)
                uploaded_feature_files[w] = {
                    "uploaded": True,
                    "ref": up_ref,
                    "meta": info,
                    "rf_top": rf_top,
                }
                continue
            except Exception as e:
                logger.warning(
                    f"[AudioBatch: {batch_label}] Failed to upload {info['json_path']}: {e}"
                )

        # fallback (preview)
        preview = make_preview_from_json(
            info["json_path"], n_frames=Constants.PREVIEW_N_FRAMES
        )
        uploaded_feature_files[w] = {
            "uploaded": False,
            "ref": None,
            "meta": info,
            "preview": preview,
            "rf_top": rf_top,
        }

    # 5) Build manifest for this batch
    manifest_entries = []
    for w, d in uploaded_feature_files.items():
        meta = d.get("meta", {})
        rf_top_str = (
            ", ".join(
                [f"{r['feature']}({r['importance']:.4f})" for r in d.get("rf_top", [])]
            )
            or "none"
        )

        if d.get("uploaded"):
            ref_name = getattr(d["ref"], "name", str(d["ref"]))
            manifest_entries.append(
                f"- audio_window_s={w:.3f}: uploaded_file={ref_name}, rf_top=[{rf_top_str}]"
            )
        else:
            preview_len = len(d.get("preview") or [])
            manifest_entries.append(
                f"- audio_window_s={w:.3f}: preview_len={preview_len}, rf_top=[{rf_top_str}]"
            )

    audio_manifest_text = (
        Constants.AUDIO_MANIFEST_HEADER.format(batch_label=batch_label)
        + "\n".join(manifest_entries)
    )

    # 6) Assemble content list
    contents_for_gemini: List[Any] = [audio_manifest_text]

    if audio_upload_ref is not None:
        contents_for_gemini.append(audio_upload_ref)
    else:
        contents_for_gemini.append(
            f"{Constants.AUDIO_FILENAME_PREFIX}{Path(audio_path).name}"
        )

    if rf_guidance_text_parts:
        contents_for_gemini.append("\n".join(rf_guidance_text_parts))
    print(uploaded_feature_files)
    for w, d in uploaded_feature_files.items():
        if d.get("uploaded"):
            print(d["ref"])
            contents_for_gemini.append(d["ref"])
        else:
            preview_text = (
                json.dumps(d.get("preview"))
                if not isinstance(d.get("preview"), str)
                else d["preview"]
            )
            contents_for_gemini.append(
                Constants.PREVIEW_PREFIX.format(w=w) + preview_text
            )

    batch_audio_meta = {
        "audio_chunk_path": audio_path,
        "audio_upload_ref": getattr(audio_upload_ref, "name", None),
        "feature_files": {
            w: (
                d["ref"].name
                if d.get("uploaded")
                else f"preview_len_{len(d.get('preview', []))}"
            )
            for w, d in uploaded_feature_files.items()
        },
        "csv_paths": [v["csv_path"] for v in per_window.values()],
    }

    logger.info(
        f"[AudioBatch: {batch_label}] Finished. Returning {len(contents_for_gemini)} content parts."
    )

    return contents_for_gemini, batch_audio_meta
