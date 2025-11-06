import json
from src.python.app.constants.constants import Constants

def build_state_summary(state: dict) -> str:
    summary = {}

    if state.get(Constants.SAMPELED_BLENDSAHPE_CSV_PATH_KEY):
        summary[Constants.CSV_SAMPLER_KEY] = f"Sampled CSV created at {state[Constants.SAMPELED_BLENDSAHPE_CSV_PATH_KEY]}"

    if state.get(Constants.PREFILTER_SUMMARY_KEY):
        summary[Constants.PREFILTER_KEY] = state[Constants.PREFILTER_SUMMARY_KEY]

    if state.get(Constants.FILTERED_BLENDSHAPE_CSV_PATH_KEY):
        summary[Constants.CSV_FILTER_KEY] = f"Filtered CSV created at {state[Constants.FILTERED_BLENDSHAPE_CSV_PATH_KEY]}"

    if state.get(Constants.TARGET_FPS_KEY):
        summary[Constants.TARGET_FPS_KEY] = state[Constants.TARGET_FPS_KEY]

    return json.dumps(summary, indent=2)