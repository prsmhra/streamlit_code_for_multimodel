from typing import Optional
from google.adk.tools import ToolContext, FunctionTool
import pandas as pd
import os
import re

from src.python.app.constants.constants import Constants


def sample_csv_tool(
    target_fps: int,
    reason: str,
    tool_context: ToolContext
) -> dict:
    """
    Downsamples the input blendshape CSV to a target FPS and saves it.
    
    Args:
        target_fps: The target frames per second after sampling.
        reason: Explanation for why this FPS was chosen.
        tool_context: Context containing csv_path and out_dir.
    """
    csv_path = tool_context.state.get("csv_path")
    out_dir = tool_context.state.get("out_dir", ".")
    input_fps = tool_context.state.get("original_fps", 30)

    if not csv_path or not os.path.exists(csv_path):
        result = {
            "success": False,
            "reason": "CSV not found",
            "sampled_csv_path": None,
            "target_fps": None
        }
        tool_context.state["csv_sampler_result"] = result
        return result

    # Read and sample
    df = pd.read_csv(csv_path)
    step = max(1, input_fps // target_fps)
    sampled_df = df.iloc[::step, :].reset_index(drop=True)
    # sampled_df["frame_no"] = range(1, len(sampled_df) + 1)

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(csv_path)
    match = re.match(r"(batch_\d+)", filename)
    if match:
        batch_name = match.group(1)
        sampled_csv_path = os.path.join(out_dir, f"{batch_name}_sampled_blendshape.csv")

    else:
        sampled_csv_path = os.path.join(out_dir, "sampled_blendshape.csv")

    sampled_df.to_csv(sampled_csv_path, index=False)

    result = {
        "success": True,
        "reason": reason,
        "sampled_csv_path": sampled_csv_path,
        "target_fps": target_fps
    }
    tool_context.state["csv_sampler_result"] = result
    return result


# Wrap it as a FunctionTool
sample_csv_function_tool = FunctionTool(func=sample_csv_tool)
