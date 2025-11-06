
import os
import re
import json
import pandas as pd
from typing import List, Dict
from google.adk.tools import ToolContext, FunctionTool

from src.python.app.constants.constants import Constants

def filter_blendshape_csv_tool(
    regions: List[str],
    tool_context: ToolContext,
):
    """
    Filters blendshape CSV to keep only relevant columns based on detected facial regions.
    
    Args:
        regions: List of facial regions to include (e.g., ["eyes", "mouth", "nose"])
        tool_context: Context containing csv_path, region_map, au_region_map, emotion_cols, out_dir
    """
    
    # Get parameters from context
    csv_path = tool_context.state.get(Constants.CSV_PATH_KEY)
    region_map = tool_context.state.get(Constants.REGION_MAP_KEY)
    au_region_map = tool_context.state.get(Constants.AU_REGION_MAP_KEY)
    emotion_cols = tool_context.state.get(Constants.EMOTION_COLS_KEY)
    out_dir = tool_context.state.get(Constants.OUT_DIR_KEY)
    
    if not csv_path or not os.path.exists(csv_path):
        result = {
            Constants.SUCCESS_KEY: False,
            Constants.REASON_KEY: "CSV not found",
            Constants.FILTERED_CSV_PATH_KEY: None,
        }
        tool_context.state[Constants.FILTERED_CSV_RESULT_KEY] = result
        return result
    
    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    match = re.match(r"(batch_\d+)", filename)
    if match:
        batch_name = match.group(1)
        filtered_path = os.path.join(out_dir, f"{batch_name}_{Constants.BLENDSHAPE_AU_EMOTION_FILTERED_KEY}")
    else:
        filtered_path = os.path.join(out_dir, f"{Constants.BLENDSHAPE_AU_EMOTION_FILTERED_KEY}")
    
    all_cols = df.columns.tolist()
    frame_col = [Constants.FRAME_KEY]
    blendshape_cols = all_cols[all_cols.index(Constants.FRAME_KEY) + 1 : all_cols.index(Constants.NOSESNEERRIGHT_KEY) + 1]
    remaining_cols = all_cols[all_cols.index(Constants.NOSESNEERRIGHT_KEY) + 1 :]
    au_cols = [c for c in remaining_cols if c not in emotion_cols]
    
    # Select columns
    keep_cols = frame_col.copy()
    for r in regions:
        keep_cols.extend([c for c in region_map.get(r, []) if c in blendshape_cols])
    for r in regions:
        keep_cols.extend([c for c in au_region_map.get(r, []) if c in au_cols])
    keep_cols.extend(emotion_cols)
    
    df_filtered = df[keep_cols]
    
    df_filtered.to_csv(filtered_path, index=False)
    
    result = {Constants.SUCCESS_KEY: True, Constants.FILTERED_CSV_PATH_KEY: filtered_path, Constants.REGIONS_KEY: regions}
    tool_context.state[Constants.FILTERED_CSV_RESULT_KEY] = result


    return result

# Wrap it as a FunctionTool
csv_filter_tool = FunctionTool(func=filter_blendshape_csv_tool)