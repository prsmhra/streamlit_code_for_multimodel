import os
import re

import pandas as pd
from typing import List
from google.adk.tools import ToolContext, FunctionTool
from src.python.app.constants.constants import Constants

def prefilter_frames_tool(
    useful: bool,
    frame_ranges: List[List[int]],
    reason: str,
    tool_context: ToolContext
) -> dict:
    """
    Filter frames from blendshape CSV based on LLM decision.
    
    Args:
        useful: Whether the data contains useful frames for medical analysis
        frame_ranges: List of [start, end] frame ranges to keep (inclusive), e.g., [[20, 50], [70, 85]]
        reason: Explanation for the decision
        tool_context: Context containing csv_path and out_dir
    """
    # Get parameters from context
    csv_path = tool_context.state.get(Constants.CSV_PATH_KEY)
    out_dir = tool_context.state.get(Constants.OUT_DIR_KEY, ".")
    
    if not csv_path or not os.path.exists(csv_path):
        result = {
            Constants.USEFUL_KEY: False,
            Constants.REASON_KEY: "CSV not found",
            Constants.FILTERED_CSV_PATH_KEY: None,
            Constants.KEPT_RANGES_KEY: []
        }
        tool_context.state[Constants.PREFILTER_RESULT_KEY] = result
        return result
    
    df = pd.read_csv(csv_path)
    
    if not useful:
        result = {
            Constants.USEFUL_KEY: False,
            Constants.REASON_KEY: reason,
            Constants.FILTERED_CSV_PATH_KEY: None,
            Constants.KEPT_RANGES_KEY: []
        }
        tool_context.state[Constants.PREFILTER_RESULT_KEY] = result
        return result
    
    keep_indices = []

    for range_pair in frame_ranges:
        if len(range_pair) == 2:
            start, end = range_pair
            keep_indices.extend(range(start, end + 1))
    
    df_filtered = df[df[Constants.FRAME_KEY].isin(keep_indices)]
    
    filename = os.path.basename(csv_path)
    match = re.match(r"(batch_\d+)", filename)
    if match:
        batch_name = match.group(1)
        filtered_path = os.path.join(out_dir, f"{batch_name}_{Constants.PREFILTER_FRAME_CSV}")
    else :
        filtered_path = os.path.join(out_dir, f"{Constants.PREFILTER_FRAME_CSV}")
    
    
    df_filtered.to_csv(filtered_path, index=False)
    
    result = {
        Constants.USEFUL_KEY: True,
        Constants.REASON_KEY: reason,
        Constants.FILTERED_CSV_PATH_KEY: filtered_path,
        Constants.KEPT_RANGES_KEY: frame_ranges
    }
    tool_context.state[Constants.PREFILTER_RESULT_KEY] = result
    return result

# Wrap it as a FunctionTool
prefilter_frames_function_tool = FunctionTool(func=prefilter_frames_tool)