import re
import json
from src.python.app.constants.constants import Constants


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*)\s*```\s*$",
                       re.DOTALL | re.IGNORECASE)

def strip_code_fence(text: str) -> str:
    if not isinstance(text, str):
        return text
    m = _FENCE_RE.match(text.strip())
    return m.group(Constants.BODY_KEY).strip() if m else text

def extract_braced_json(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -Constants.ONE or end == -Constants.ONE or end < start:
        return None
    return text[start:end+Constants.ONE]

def to_text(value) -> str:
    if value is None:
        return Constants.NOT_ASSIGN_KEY
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)

def normalize_per_speaker(per_speaker_raw) -> list[dict]:
    """
    Normalize per_speaker_findings into:
      [{"speaker": str, "conclusions": str, "evidence": str}, ...]
    Supports dict-of-dict, dict-of-str, list (mixed), or a single string.
    """
    normalized: list[dict] = []

    if isinstance(per_speaker_raw, dict):
        for speaker, info in per_speaker_raw.items():
            if isinstance(info, dict):
                normalized.append({
                    Constants.SPEAKER_KEY: speaker,
                    Constants.CONCLUSION_KEY: to_text(info.get(Constants.CONCLUSION_KEY)),
                    Constants.EVIDENCE_KEY: to_text(info.get(Constants.EVIDENCE_KEY)),
                })
            else:
                normalized.append({
                    Constants.SPEAKER_KEY: speaker,
                    Constants.CONCLUSION_KEY: to_text(info),
                    Constants.EVIDENCE_KEY: Constants.NOT_ASSIGN_KEY,
                })
        return normalized

    if isinstance(per_speaker_raw, list):
        for item in per_speaker_raw:
            if isinstance(item, dict):
                normalized.append({
                    Constants.SPEAKER_KEY: to_text(item.get(Constants.SPEAKER_KEY, Constants.UNKNOWN_KEY)),
                    Constants.CONCLUSION_KEY: to_text(item.get(Constants.CONCLUSION_KEY)),
                    Constants.EVIDENCE_KEY: to_text(item.get(Constants.EVIDENCE_KEY)),
                })
            else:  # string/other
                normalized.append({
                    Constants.SPEAKER_KEY: Constants.UNKNOWN_KEY,
                    Constants.CONCLUSION_KEY: to_text(item),
                    Constants.EVIDENCE_KEY: Constants.NOT_ASSIGN_KEY,
                })
        return normalized

    if isinstance(per_speaker_raw, str):
        return [{Constants.SPEAKER_KEY: Constants.UNKNOWN_KEY, 
                 Constants.CONCLUSION_KEY: to_text(per_speaker_raw), 
                 Constants.EVIDENCE_KEY: Constants.NOT_ASSIGN_KEY}]

    return []

def normalize_evidence_list(evidence_raw) -> list[dict]:
    """
    Normalize evidence_list to safe dicts with defaults.
    """
    if not isinstance(evidence_raw, list):
        return []
    norm = []
    for item in evidence_raw:
        if not isinstance(item, dict):
            norm.append({
                Constants.SPEAKER_KEY: Constants.UNKNOWN_KEY,
                Constants.WINDOW_LEN_S_KEY: Constants.NOT_ASSIGN_KEY,
                Constants.FRAME_IDX_KEY: Constants.NOT_ASSIGN_KEY,
                Constants.FEATURES_KEY.lower(): to_text(item),
                Constants.TOP_RF_FEATURES_KEY: Constants.NOT_ASSIGN_KEY,
                Constants.ACOUSTIC_CLAIM_KEY: Constants.NOT_ASSIGN_KEY,
            })
            continue
        norm.append({
            Constants.SPEAKER_KEY: to_text(item.get(Constants.SPEAKER_KEY, Constants.UNKNOWN_KEY)),
            Constants.WINDOW_LEN_S_KEY: item.get(Constants.WINDOW_LEN_S_KEY, Constants.NOT_ASSIGN_KEY),
            Constants.FRAME_IDX_KEY: item.get(Constants.FRAME_IDX_KEY, Constants.NOT_ASSIGN_KEY),
            Constants.FEATURES_KEY.lower(): item.get(Constants.FEATURES_KEY.lower(), {}),
            Constants.TOP_RF_FEATURES_KEY: to_text(item.get(Constants.TOP_RF_FEATURES_KEY, Constants.INVERTED_STRING)),
            Constants.ACOUSTIC_CLAIM_KEY: to_text(item.get(Constants.ACOUSTIC_CLAIM_KEY, Constants.INVERTED_STRING)),
        })
    return norm


