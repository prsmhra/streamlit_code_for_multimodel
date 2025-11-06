import logging
import os
from typing import Optional, Tuple

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import all agents and instructions
from .agents import (
    # MetaIntentAgent is NO LONGER USED HERE
    CSVSamplerAgent,
    FramePrefilterAgent,
    CSVFilterAgent,
    LlmOrchestratorAgent
)
from .instructions import (
    # META_INTENT_INSTRUCTION is NO LONGER USED HERE
    CSV_SAMPLER_INSTRUCTION,
    PREFILTER_INSTRUCTION,
    REGION_DETECTOR_INSTRUCTION,
    ORCHESTRATOR_INSTRUCTION
)
from Config.config import MODEL_NAME
from src.python.app.constants.constants import Constants
logger = logging.getLogger(__name__)


# --- Setup function for Main Pipeline ---
async def setup_vision_pipeline_runner(
    app_name: str
) -> Tuple[Runner, InMemorySessionService]:
    """Sets up the main ADK processing pipeline."""
    
    session_service = InMemorySessionService()
    
    # 1. Sampler Agent
    sample_llm_agent = LlmAgent(
        name=Constants.SAMPLINGFRAME_AGENT, model=MODEL_NAME,
        instruction=CSV_SAMPLER_INSTRUCTION, output_key=Constants.CSV_SAMPLER_RESULT_KEY
    )
    sample_frame_agent = CSVSamplerAgent(sample_llm_agent)

    # 2. Prefilter Agent
    prefilter_llm = LlmAgent(
        name=Constants.PREFILTER_AGENT, model=MODEL_NAME,
        instruction=PREFILTER_INSTRUCTION, output_key=Constants.PREFILTER_DECISION_KEY
    )
    prefilter_agent = FramePrefilterAgent(prefilter_llm)

    # 3. Feature/Region Filter Agent
    region_llm_agent = LlmAgent(
        name=Constants.REGIONDETECTOR_AGENT, model=MODEL_NAME,
        instruction=REGION_DETECTOR_INSTRUCTION, output_key=Constants.ACTIVE_REGION_KEY
    )
    csv_filter_agent = CSVFilterAgent(region_llm_agent)
    
    # 4. Orchestrator Agent
    orchestrator_llm = LlmAgent(
        name=Constants.ORCHESTRATOR_AGENT, model=MODEL_NAME,
        instruction=ORCHESTRATOR_INSTRUCTION, output_key=Constants.ORCHESTRATOR_KEY
    )
    orchestrator = LlmOrchestratorAgent(
        orchestrator_llm,
        tools=[sample_frame_agent, prefilter_agent, csv_filter_agent]
    )
    
    runner = Runner(agent=orchestrator, app_name=app_name, session_service=session_service)
    return runner, session_service

# --- Main Entry Point Function ---
async def run_vision_pipeline_for_batch(
    disease_focus: Optional[str],
    input_csv_path: str,
    output_dir: str,
    batch_label: str,
    app_name: str = "vision_batch_app",
    user_id: str = "batch_user"
) -> Optional[str]:
    """
    Runs the vision processing pipeline (Sample -> Prefilter -> Filter).
    Meta-intent is now handled by the main orchestrator.

    Returns:
        The file path to the final filtered CSV, or None if it fails.
    """
    
    
    # --- STAGE 2: MAIN PROCESSING PIPELINE ---
    logger.info(f"[VisionBatch: {batch_label}] Starting main processing pipeline...")
    pipeline_runner, pipeline_session_srv = await setup_vision_pipeline_runner(app_name)
    pipeline_session_id = f"processing_session_{batch_label}"
    
    batch_output_dir = output_dir 
    os.makedirs(batch_output_dir, exist_ok=True)
    
    INITIAL_STATE = {
        Constants.BLENDSHAPE_CSV_PATH_KEY: input_csv_path,
        Constants.WORK_DIR_KEY: batch_output_dir,
        Constants.DISEASE_FOCUS_KEY: disease_focus, # Pass the focus from main orchestrator
        Constants.BATCH_LABEL_KEY: batch_label
    }
    
    pipeline_session = await pipeline_session_srv.create_session(
        app_name=app_name, user_id=user_id, session_id=pipeline_session_id, state=INITIAL_STATE.copy()
    )
    
    # Run the processing pipeline
    async for ev in pipeline_runner.run_async(
        user_id=user_id,
        session_id=pipeline_session_id,
        new_message=types.Content(role=Constants.USER_ROLE, parts=[types.Part(text="Start processing.")])
    ):
       if ev.content and ev.content.parts:
           logger.info(f"[{ev.author}] {ev.content.parts[0].text}")

    # Get the final result
    final_state = pipeline_session_srv.get_session_sync(
        app_name=app_name, user_id=user_id, session_id=pipeline_session_id
    ).state


    filtered_csv_path = None
    filtered_result_dict = final_state.get(Constants.FILTERED_CSV_RESULT_KEY)
    
    if filtered_result_dict and isinstance(filtered_result_dict, dict):
        filtered_csv_path = filtered_result_dict.get(Constants.FILTERED_CSV_PATH_KEY)
    
    if not filtered_csv_path:
        logger.error(f"[VisionBatch: {batch_label}] Pipeline finished but final filtered CSV path was not found.")
        return None,disease_focus

    logger.info(f"[VisionBatch: {batch_label}] Pipeline finished. Final CSV: {filtered_csv_path}")
    
    return filtered_csv_path,disease_focus