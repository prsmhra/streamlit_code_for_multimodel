import json
from src.python.app.constants.constants import Constants

def build_state_summary(state: dict) -> str:
    summary = {}

    if state.get("prefilter_summary"):
        summary["prefilter"] = state["prefilter_summary"]

    if state.get("filtered_blendshape_csv_path"):
        summary["csv_filter"] = f"Filtered CSV created at {state['filtered_blendshape_csv_path']}"


    if state.get("sampled_blendshape_csv_path"):
        summary["csv_sampler"] = f"Sampled CSV created at {state['sampled_blendshape_csv_path']}"

    if state.get("target_fps"):
        summary["target_fps"] = state["target_fps"]

    if state.get("symptom_analysis"):
        raw = state["symptom_analysis"]
        summary["symptom_analysis"] = raw[:Constants.FIVE_HUNDERD] + ("..." if len(raw) > Constants.FIVE_HUNDERD else Constants.INVERTED_STRING)

    # Include whether sample data is provided
    if state.get("meta_intent_result"):
        try:
            meta = state["meta_intent_result"]
            if isinstance(meta, dict) and meta.get("sample_data_provided"):
                summary["sample_data_provided"] = True
        except Exception:
            pass

    return json.dumps(summary, indent=2)