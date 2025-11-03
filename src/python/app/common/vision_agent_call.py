import os
import re
import io
import ast
import json
import random
import logging
import asyncio

import pandas as pd
import streamlit as st
from datetime import datetime
from google.genai import types
from dotenv import load_dotenv
from pydantic import PrivateAttr
import plotly.graph_objects as go

from collections import defaultdict
from google.adk.events import Event
from google.adk.runners import Runner
from typing_extensions import override

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.sessions import InMemorySessionService

from Config import config
from src.python.app.tools.frame_prefilter_tools import prefilter_frames_function_tool
from src.python.app.tools.csv_filter_tools import csv_filter_tool
from src.python.app.tools.sample_function_tools import sample_csv_function_tool
from src.python.app.video_frame_extractor.csv_sav_inference import Infer  

# from src.ui.diagram import create_agent_diagram
from src.python.app.utils.get_list_from_str import get_list_from_str
from src.python.app.constants.constants import Constants
from src.python.app.common.vision_agents import (
    MetaIntentAgent, CSVSamplerAgent, FramePrefilterAgent,
    CSVFilterAgent, SymptomAnalyzerAgent, LlmOrchestratorAgent
)
from src.python.app.instructions.vision_agent_instructions import (
    META_INTENT_INSTRUCTION, CSV_SAMPLER_INSTRUCTION, PREFILTER_INSTRUCTION,
    REGION_DETECTOR_INSTRUCTION, SYMPTOM_ANALYZER_INSTRUCTION, ORCHESTRATOR_INSTRUCTION
)

from src.python.app.utils.batching import create_batches, format_batches_for_summary

from src.python.app.utils.data_utils import (
    detect_csv_structure, load_data_from_sources, get_region_columns,
    extract_frames_from_message, parse_prefilter_decision, get_prefiltered_frames,
    group_logs_by_batch
)

from src.python.app.common.vision_single_batch_runner import run_pipeline_for_batch_async, run_summary_agent_async

from src.python.app.utils.ui_renders import (
    render_emotions_pain, render_regional_aus, render_regional_blendshapes,
    render_regional_comparison, render_frame_range_detailed_analysis,
    render_prefilter_visualization, render_batch_section, render_log_card
)

# --------------------------------------------------
# 🔧 Setup logging + env
# --------------------------------------------------
logger = config.get_logger(__name__)

# INPUT_FPS = config.INPUT_FPS
# APP_NAME = config.APP_NAME
# USER_ID = config.USER_ID
# MODEL_NAME = config.MODEL_NAME


class MedicalAIAgentApp:
    """
    A class to encapsulate the Medical AI Agent Pipeline.
    NOTE: This class does NOT set page config when used as a module.
    Page config is handled by inference.py (the entry point).
    """

    def __init__(self):
        """
        Initialize the application - NO page config here when used as module
        """
        self.work_dir = Constants.VISION_OUT_DIR
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initializes all required keys in the Streamlit session state."""
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = []
        if 'final_summary' not in st.session_state:
            st.session_state.final_summary = ""
        if 'current_batch_statuses' not in st.session_state:
            st.session_state.current_batch_statuses = {}
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        
        if not st.session_state.current_batch_statuses:
            self._reset_statuses()

    def _reset_statuses(self):
        """Helper to reset agent statuses for the diagram."""
        st.session_state.current_batch_statuses = {
            "LlmOrchestrator": "not_started",
            "MetaIntentTool": "not_started",
            "FrameSamplerTool": "not_started",
            "FramePrefilterTool": "not_started",
            "FeaturesSelectionTool": "not_started",
            "SymptomAnalyzerTool": "not_started"
        }

    async def _main_pipeline_async(self):
        """
        The main asynchronous function that orchestrates the batch processing.
        """
        session_service = InMemorySessionService()
        stop_processing = False
        
        # Set processing state
        st.session_state.currently_processing = True
        st.session_state.expected_total_batches = self.total_batches

        for i, batch_df in enumerate(self.batches):
            batch_num = i + Constants.ONE
            
            # Update current batch being processed
            st.session_state.current_batch_index = i
            
            self._reset_statuses()
            st.session_state.current_batch_statuses["LlmOrchestrator"] = "running"
            self.status_text.info(f"☑️ Preparing Batch {batch_num} of {self.total_batches}...")
            
            await asyncio.sleep(1.5)                 
            self.status_text.info(f"🚀 **Processing Batch {batch_num} of {self.total_batches}...** ({len(batch_df)} frames)")
            
            batch_path = os.path.join(self.work_dir, f"batch_{i}.csv")
            batch_df.to_csv(batch_path, index=False)

            current_batch_logs = []
            event_count_in_batch = 0
            session_id = f"session_batch_{i}_{datetime.now().strftime('%H%M%S')}"
            
            async for event, session_state in run_pipeline_for_batch_async(
                batch_path, self.work_dir, self.user_prompt, session_service, session_id
            ):
                author = event.author or "System"

                # Update statuses based on agent activity
                if "MetaIntent" in author: 
                    st.session_state.current_batch_statuses["MetaIntentTool"] = "running"
                    await asyncio.sleep(1.5)
                elif "SamplingFrames" in author:
                    st.session_state.current_batch_statuses["FrameSamplerTool"] = "running"
                elif "Prefilter" in author:
                    st.session_state.current_batch_statuses["FramePrefilterTool"] = "running"
                elif "FeaturesSelection" in author or "Region" in author:
                    st.session_state.current_batch_statuses["FeaturesSelectionTool"] = "running"
                elif "Symptom" in author:
                    st.session_state.current_batch_statuses["SymptomAnalyzerTool"] = "running"

                if event.content and event.content.parts:
                    message = " ".join([p.text.strip() for p in event.content.parts if p.text])
                    
                    # Check for completion markers
                    if "✅" in message:
                        if "MetaIntent analysis" in message:
                            st.session_state.current_batch_statuses["MetaIntentTool"] = "completed"
                            # Check for invalid input (only on first batch)
                            if i == 0:
                                match = re.search(r"(\{.*\})", message, re.DOTALL)
                                if match:
                                    try:
                                        intent_data = ast.literal_eval(match.group(1))
                                        if intent_data.get("intent_type") == "invalid_input":
                                            self._reset_statuses()
                                            error_message = (
                                                "❌ Execution stopped. This application is for facial medical analysis only. "
                                                f"Reason: {intent_data.get('reason', 'The prompt was not related to this tool.')}"
                                            )
                                            self.status_text.error(error_message)
                                            st.warning(error_message)
                                            stop_processing = True
                                            break
                                    except (json.JSONDecodeError, ValueError, SyntaxError):
                                        pass
                                        
                        elif "CSV sampled" in message:
                            st.session_state.current_batch_statuses["FrameSamplerTool"] = "completed"
                        elif "Prefilter kept frames" in message:
                            st.session_state.current_batch_statuses["FramePrefilterTool"] = "completed"
                        elif "Filtered CSV saved" in message:
                            st.session_state.current_batch_statuses["FeaturesSelectionTool"] = "completed"
                        elif "Symptom analysis process completed" in message:
                            st.session_state.current_batch_statuses["SymptomAnalyzerTool"] = "completed"
                        elif "Orchestrator finished flow" in message:
                            st.session_state.current_batch_statuses["LlmOrchestrator"] = "completed"

                    event_count_in_batch += Constants.ONE
                    
                    # Update status and logs
                    self.status_text.info(f"**[Batch {batch_num}]** [{author}] {message[:150]}...")
                    with self.log_container:
                        # Enhanced log styling with animation
                        log_class = "processing-log" if "running" in message.lower() or "processing" in message.lower() else ""
                        st.markdown(
                            f'<div class="agent-log-card {author} {log_class}">'
                            f'<div class="agent-log-header">'
                            f'<span class="batch-badge" style="background: rgba(106, 13, 173, 0.2); color: #6a0dad;">Batch {batch_num}</span>'
                            f'<span class="agent-name">{author}</span>'
                            f'</div>'
                            f'<div class="agent-message" style="color: #585a5e;">{message}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    current_batch_logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "agent": author,
                        "message": message
                    })

            # If invalid input detected, stop processing
            if stop_processing:
                self._reset_statuses()
                st.session_state.currently_processing = False
                return 
            
            # Retrieve final session state
            final_session = await session_service.get_session(
                app_name=Constants.APP_NAME,
                user_id=Constants.USER_ID,
                session_id=session_id
            )
            
            # Store batch results
            st.session_state.batch_data.append({
                "batch_id": i,
                "agent_logs": current_batch_logs,
                "session_state_data": dict(final_session.state),
                "original_df": batch_df 
            })
            
            self.status_text.success(f"✅ **Batch {batch_num} of {self.total_batches} complete!**")
            self.progress_bar.progress(int((batch_num / self.total_batches) * 100))
            await asyncio.sleep(Constants.ONE) 
            
        # Generate final summary
        self.status_text.info("📊 **Generating final summary across all batches...**")
        summary_input = format_batches_for_summary(st.session_state.batch_data)
        
        async for event in run_summary_agent_async(summary_input, session_service):
            if event.content and event.content.parts:
                st.session_state.final_summary = event.content.parts[0].text

        # Mark processing complete
        st.session_state.processing_complete = True
        st.session_state.currently_processing = False
        
        self.status_text.success("🎉 **All Batches Processed and Summarized!**")
        self._reset_statuses()
        
        # Force rerun to show results
        st.rerun()

    def _run_processing_pipeline(self, df_check, work_dir, batch_size, user_prompt):
        """
        Handles the setup and execution of the asynchronous processing pipeline.
        
        Args:
            input_file: Streamlit UploadedFile object (video or CSV)
            work_dir: Directory where outputs will be saved
            batch_size: Number of frames per batch
            user_prompt: User's analysis prompt
        """
        try:
            os.makedirs(work_dir, exist_ok=True)
            
            # Determine input type
            # file_name = input_file.name
            # file_ext = file_name.split(Constants.DOT)[-Constants.ONE].lower()
            # if file_ext in Constants.VIDEO_EXT:
            #     # Video processing
            #     video_path = os.path.join(work_dir, "input_video.mp4")
            #     with open(video_path, Constants.WRITE_BINARY) as f:
            #         f.write(input_file.getvalue())
                
            #     st.info("🎬 Video processing: Extracting blendshapes...")
            #     csv_path = Infer(video_path).inference()
            #     df_check = pd.read_csv(csv_path)
                
            # elif file_ext in Constants.CSV_EXT:
            #     # CSV processing
            #     csv_path = os.path.join(work_dir, "input_data.csv")
            #     with open(csv_path, Constants.WRITE_BINARY) as f:
            #         f.write(input_file.getvalue())
            #     df_check = pd.read_csv(csv_path)
                
            # else:
            #     st.error(f"Unsupported file type: {file_ext}")
            #     return
            
            # Check for AUs
            has_aus = any('AU' in col for col in df_check.columns)
            if has_aus:
                st.info("✅ CSV contains AUs - proceeding.")
            else:
                st.info("📊 CSV contains only blendshapes - AUs will be calculated.")
            
            st.session_state.original_blendshapes_df = df_check
            
            # Create batches
            self.batches = create_batches(df_check, batch_size)
            self.total_batches = len(self.batches)
            self.work_dir = work_dir
            self.user_prompt = user_prompt
            
            st.info(f"📦 Created {self.total_batches} batches with ~{batch_size} frames each")
            
            # Create UI placeholders
            self.diagram_placeholder = st.empty()
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.log_container = st.container()

            # Clear previous results
            st.session_state.batch_data = []
            st.session_state.final_summary = ""
            
            # Run the main async pipeline
            asyncio.run(self._main_pipeline_async())

        except Exception as e:
            st.error(f"❌ An error occurred during pipeline execution: {e}")
            logger.error("Pipeline failed", exc_info=True)
            st.session_state.processing_complete = False

    def run(self):
        """
        Standalone run method - ONLY use if running vision_agent_call.py directly.
        DO NOT use when called from web_ui.py
        """
        # This is for standalone execution only
        st.set_page_config(
            page_title="Medical AI Agent Pipeline",
            page_icon="🥼",
            layout="wide"
        )
        
        st.title("🥼 Medical AI Agent Pipeline")
        st.markdown("Automated facial analysis using Google ADK agents")
        st.markdown("---")
        
        # Simple file uploader
        input_file = st.file_uploader("Upload CSV or Video", type=['csv', 'mp4', 'avi', 'mov', 'mkv'])
        work_dir = st.text_input("Output Directory", value="./output")
        batch_size = st.number_input("Batch Size (frames)", min_value=50, max_value=500, value=100, step=50)
        user_prompt = st.text_area("Analysis Prompt", value="Analyze this data for medical assessment")
        
        if st.button("🚀 Start Analysis") and input_file:
            self._run_processing_pipeline(input_file, work_dir, batch_size, user_prompt)
        
        # Show results if processing complete
        if st.session_state.processing_complete and st.session_state.batch_data:
            st.markdown("---")
            st.header("📊 Results")
            
            for i, batch in enumerate(st.session_state.batch_data):
                with st.expander(f"Batch {i+1}"):
                    st.json(batch['session_state_data'])
            
            if st.session_state.final_summary:
                st.markdown("### Final Summary")
                st.markdown(st.session_state.final_summary)


# Standalone execution
if __name__ == "__main__":
    app = MedicalAIAgentApp()
    app.run()