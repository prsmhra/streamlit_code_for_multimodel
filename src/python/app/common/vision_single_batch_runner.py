import os
import re
import io
import gc
import ast
import json
import random
import logging
import asyncio
import aiohttp



from datetime import datetime

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.api_core import exceptions
from google.genai.errors import ClientError 

from src.python.app.constants.constants import Constants
from src.python.app.tools.frame_prefilter_tools import prefilter_frames_function_tool
from src.python.app.tools.csv_filter_tools import csv_filter_tool
from src.python.app.tools.sample_function_tools import sample_csv_function_tool



from src.python.app.common.vision_agents import (
    MetaIntentAgent, CSVSamplerAgent, FramePrefilterAgent,
    CSVFilterAgent, SymptomAnalyzerAgent, LlmOrchestratorAgent
)
from src.python.app.instructions.agent_instructions import (
    META_INTENT_INSTRUCTION, CSV_SAMPLER_INSTRUCTION, PREFILTER_INSTRUCTION,
    REGION_DETECTOR_INSTRUCTION, SYMPTOM_ANALYZER_INSTRUCTION, ORCHESTRATOR_INSTRUCTION
)
from Config import config 
logger = config.get_logger(__name__)
APP_NAME = config.APP_NAME
USER_ID = config.USER_ID
MODEL_NAME = config.MODEL_NAME

from src.python.app.utils.batching import create_batches,format_batches_for_summary



async def safe_run_runner(runner, user_id, session_id, message, batch_index):
    """
    Run runner with retry and exponential backoff on quota or transient errors.
    Automatically closes aiohttp sessions if errors occur.
    """
    max_retries = Constants.FIVE
    for attempt in range(max_retries):
        try:
            async for ev in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message
            ):
                yield ev
            return  # ✅ success, stop retries

        except ClientError as e:
            code = getattr(e, "status_code", None)
            if code in (429, 503):
                delay = 15 + random.uniform(0, Constants.FIVE) # Use a fixed delay with jitter
                logger.warning(f"[Batch {batch_index}] ⚠️ API Error {code}. Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"[Batch {batch_index}] ❌ ClientError: {e}")
                raise

        except Exception as e:
            logger.error(f"[Batch {batch_index}] ❌ An unexpected error occurred: {e}")
            raise

        finally:
            # ✅ Clean up dangling aiohttp sessions to silence warnings
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                    await obj.close()

    raise RuntimeError(f"Pipeline failed for Batch {batch_index} after {max_retries} retries.")


async def run_pipeline_for_batch_async(csv_path, work_dir, prompt, session_service, session_id):
    """Runs the orchestrated pipeline for a single batch."""
    initial_state = {"blendshape_csv_path": csv_path, "work_dir": work_dir}
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state=initial_state.copy()
    )
    
    # Agent definitions (as in your batch script)
    intent_llm_agent = LlmAgent(name="MetaIntentLLM", model=MODEL_NAME, instruction=META_INTENT_INSTRUCTION, output_key="meta_intent_result")
    meta_intent_agent = MetaIntentAgent(intent_llm_agent)
    prefilter_llm = LlmAgent(name="PrefilterLLM", model=MODEL_NAME, instruction=PREFILTER_INSTRUCTION, output_key="prefilter_decision")
    prefilter_agent = FramePrefilterAgent(prefilter_llm)
    region_llm_agent = LlmAgent(name="RegionDetectorLLM", model=MODEL_NAME, instruction=REGION_DETECTOR_INSTRUCTION, output_key="active_regions")
    csv_filter_agent = CSVFilterAgent(region_llm_agent) # Name is FeaturesSelectionTool
    sample_llm_agent = LlmAgent(name="SamplingFramesLLM", model=MODEL_NAME, instruction=CSV_SAMPLER_INSTRUCTION, output_key="csv_sampler_result")
    sample_frame_agent = CSVSamplerAgent(sample_llm_agent)
    symptom_llm_agent = LlmAgent(name="SymptomAnalyzerLLM", model=MODEL_NAME, instruction=SYMPTOM_ANALYZER_INSTRUCTION, output_key="symptom_analysis")
    symptom_agent = SymptomAnalyzerAgent(symptom_llm_agent)
    orchestrator_llm = LlmAgent(name="PipelineOrchestratorLLM", model=MODEL_NAME, instruction=ORCHESTRATOR_INSTRUCTION, output_key="orchestrator_decision")
    orchestrator = LlmOrchestratorAgent(
        orchestrator_llm,
        tools=[prefilter_agent, symptom_agent, csv_filter_agent, meta_intent_agent, sample_frame_agent]
    )
    runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service)
    
    # Yield events from the safe runner
    async for ev in safe_run_runner(
        runner=runner, user_id=USER_ID, session_id=session_id,
        message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        batch_index=session_id.split('_')[2] # Extract batch index for logging
    ):
        session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        yield ev, session.state

# This function runs the final summary agent.
async def run_summary_agent_async(summary_input, session_service):
    """Runs the final summary agent."""
    summary_session_id = f"summary_session_{datetime.now().strftime('%H%M%S')}"
    summary_instruction = f"""
    You are a senior clinical analyst AI. Summarize the following analyses from multiple batches of facial motion data.
    Highlight patterns across batches, recurring asymmetries, or consistent facial weakness. Conclude with the overall likely diagnosis or observation (if any).
    Input analyses:\n{summary_input}
    """
    summary_llm = LlmAgent(name="SummaryLLM", model=MODEL_NAME, instruction=summary_instruction, output_key="final_summary")
    runner = Runner(agent=summary_llm, app_name=APP_NAME, session_service=session_service)
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=summary_session_id, state={})
    
    async for ev in safe_run_runner(
        runner, USER_ID, summary_session_id,
        types.Content(role="user", parts=[types.Part(text="Summarize all batch analyses into one report.")]),
        "summary"
    ):
        yield ev

