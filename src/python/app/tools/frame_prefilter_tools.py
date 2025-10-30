from typing import List
from google.adk.tools import ToolContext, FunctionTool
import os
import pandas as pd
import re

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
    csv_path = tool_context.state.get("csv_path")
    out_dir = tool_context.state.get("out_dir", ".")
    
    if not csv_path or not os.path.exists(csv_path):
        result = {
            "useful": False,
            "reason": "CSV not found",
            "filtered_csv_path": None,
            "kept_ranges": []
        }
        tool_context.state["prefilter_result"] = result
        return result
    
    df = pd.read_csv(csv_path)
    
    if not useful:
        result = {
            "useful": False,
            "reason": reason,
            "filtered_csv_path": None,
            "kept_ranges": []
        }
        tool_context.state["prefilter_result"] = result
        return result
    
    keep_indices = []
    # print(frame_ranges)
    for range_pair in frame_ranges:
        if len(range_pair) == 2:
            start, end = range_pair
            keep_indices.extend(range(start, end + 1))
    
    df_filtered = df[df["frame"].isin(keep_indices)]
    
    filename = os.path.basename(csv_path)
    match = re.match(r"(batch_\d+)", filename)
    if match:
        batch_name = match.group(1)
        filtered_path = os.path.join(out_dir, f"{batch_name}_prefiltered_frames.csv")
    else :
        filtered_path = os.path.join(out_dir, f"prefiltered_frames.csv")
    
    # filtered_path = os.path.join(out_dir, f"{batch_name}_prefiltered_frames.csv"
    df_filtered.to_csv(filtered_path, index=False)
    
    result = {
        "useful": True,
        "reason": reason,
        "filtered_csv_path": filtered_path,
        "kept_ranges": frame_ranges
    }
    tool_context.state["prefilter_result"] = result
    return result

# Wrap it as a FunctionTool
prefilter_frames_function_tool = FunctionTool(func=prefilter_frames_tool)