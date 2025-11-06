

import os
import re
import pandas as pd
from typing import Optional
from google.adk.tools import ToolContext, FunctionTool

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
    csv_path = tool_context.state.get(Constants.CSV_PATH_KEY)
    out_dir = tool_context.state.get(Constants.OUT_DIR_KEY, ".")
    input_fps = tool_context.state.get(Constants.ORIGINAL_FPS_KEY, 30)

    if not csv_path or not os.path.exists(csv_path):
        result = {
            Constants.SUCCESS_KEY: False,
            Constants.REASON_KEY: "CSV not found",
            Constants.SAMPELED_CSV_PATH_KEY: None,
            Constants.TARGET_FPS_KEY: None
        }
        tool_context.state[Constants.CSV_SAMPLER_RESULT_KEY] = result
        return result

    # Read and sample
    df = pd.read_csv(csv_path)
    step = max(1, input_fps // target_fps)
    sampled_df = df.iloc[::step, :].reset_index(drop=True)
    
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(csv_path)
    match = re.match(r"(batch_\d+)", filename)
    if match:
        batch_name = match.group(1)
        sampled_csv_path = os.path.join(out_dir, f"{batch_name}_{Constants.SAMPELED_BLENDSAHPE_CSV}")

    else:
        sampled_csv_path = os.path.join(out_dir, Constants.SAMPELED_BLENDSAHPE_CSV)

    sampled_df.to_csv(sampled_csv_path, index=False)

    result = {
        Constants.SUCCESS_KEY: True,
        Constants.REASON_KEY: reason,
        Constants.SAMPELED_CSV_PATH_KEY: sampled_csv_path,
        Constants.TARGET_FPS_KEY: target_fps
    }
    tool_context.state[Constants.CSV_SAMPLER_RESULT_KEY] = result
    return result


# Wrap it as a FunctionTool
sample_csv_function_tool = FunctionTool(func=sample_csv_tool)
 