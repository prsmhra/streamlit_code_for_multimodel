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
# from src.python.app.common.csv_sav_inference import Infer  

# from src.ui.diagram import create_agent_diagram
from src.python.app.utils.get_list_from_str import get_list_from_str
from src.python.app.constants.constants import Constants
from src.python.app.common.vision_agents import (
    MetaIntentAgent, CSVSamplerAgent, FramePrefilterAgent,
    CSVFilterAgent, SymptomAnalyzerAgent, LlmOrchestratorAgent
)
from src.python.app.instructions.agent_instructions import (
    META_INTENT_INSTRUCTION, CSV_SAMPLER_INSTRUCTION, PREFILTER_INSTRUCTION,
    REGION_DETECTOR_INSTRUCTION, SYMPTOM_ANALYZER_INSTRUCTION, ORCHESTRATOR_INSTRUCTION
)



from src.python.app.utils.batching import create_batches,format_batches_for_summary

from src.python.app.utils.data_utils import (detect_csv_structure,load_data_from_sources,get_region_columns,extract_frames_from_message,
                                         parse_prefilter_decision,get_prefiltered_frames,group_logs_by_batch)

from src.python.app.common.vision_single_batch_runner import run_pipeline_for_batch_async,run_summary_agent_async

from src.python.app.utils.ui_renders import (render_emotions_pain,render_regional_aus,render_regional_blendshapes,render_regional_comparison,
                                         render_frame_range_detailed_analysis,render_prefilter_visualization,render_batch_section,render_log_card)

# --------------------------------------------------
# 🔧 Setup logging + env
# --------------------------------------------------
logger = config.get_logger(__name__)

INPUT_FPS=config.INPUT_FPS
APP_NAME = config.APP_NAME
USER_ID = config.USER_ID
MODEL_NAME = config.MODEL_NAME



class MedicalAIAgentApp:
    """
    A class to encapsulate the Medical AI Agent Pipeline Streamlit application.
    """

    def __init__(self):
        """
        Initialize the application, set page config, and inject CSS.
        """
        self.work_dir =config.DEFAULT_WORK_DIR
        # self._set_page_config()
        # self._inject_css()
        # self._initialize_session_state()

    # def _set_page_config(self):
    #     """Sets the Streamlit page configuration."""
    #     st.set_page_config(
    #         page_title="Medical AI Agent Pipeline",
    #         page_icon="🏥",
    #         layout="wide"
    #     )

    # def _inject_css(self):
    #     """Injects custom CSS styles into the Streamlit app."""
    #     st.markdown("""
    #     <style>
    #         .agent-card {
    #             padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    #             border-left: 4px solid;
    #         }
    #         .orchestrator { border-color: #9C27B0; background: #F3E5F5; }
    #         .prefilter { border-color: #2196F3; background: #E3F2FD; }
    #         .csvfilter { border-color: #FF9800; background: #FFF3E0; }
    #         .symptom { border-color: #4CAF50; background: #E8F5E9; }
    #         .metaintent { border-color: #00BCD4; background: #E0F7FA; }
    #         .csvsampler { border-color: #795548; background: #EFEBE9; }
    #         .running { animation: pulse 2s infinite; }
    #         .MetaIntentLLM {background: #FFE5E5; border: #FF6B6B; text: #8B0000;}
    #         .MetaIntentTool {background: #FFE5E5; border: #FF6B6B; text: #8B0000;}
    #         .PipelineOrchestratorLLM {background: #E5F3FF; border: #4DABF7; text: #0B3BA1;}
    #         .LlmOrchestrator {background: #E5F3FF; border: #4DABF7; text: #0B3BA1;}
    #         .FrameSamplerTool {background: #E5FCFF; border: #74C0FC; text: #004E89;}
    #         .SamplingFramesLLM {background: #E5FCFF; border: #74C0FC; text: #004E89;}
    #         .FramePrefilterTool {background: #F0E5FF; border: #B197FC; text: #5F00B2;}
    #         .PrefilterLLM {background: #F0E5FF; border: #B197FC; text: #5F00B2;}
    #         .FeaturesSelectionTool {background: #FFF4E5; border: #FFD43B; text: #995A00;}
    #         .RegionDetectorLLM {background: #FFF4E5; border: #FFD43B; text: #995A00;}
    #         .SymptomAnalyzerTool {background: #E5F5F0; border: #51CF66; text: #0B5F0B;}
    #         .SymptomReasonerLLM {background: #E5F5F0; border: #51CF66; text: #0B5F0B;}
    #         @keyframes pulse {
    #             0%, 100% { opacity: 1; } 50% { opacity: 0.7; }
    #         }
    #     </style>
    #     """, unsafe_allow_html=True)

    # def _initialize_session_state(self):
    #     """Initializes all required keys in the Streamlit session state."""
    #     if 'batch_data' not in st.session_state:
    #         st.session_state.batch_data = []
    #     if 'final_summary' not in st.session_state:
    #         st.session_state.final_summary = ""
    #     if 'current_batch_statuses' not in st.session_state:
    #         st.session_state.current_batch_statuses = {}
    #     if 'processing_complete' not in st.session_state:
    #         st.session_state.processing_complete = False
        
    #     if not st.session_state.current_batch_statuses:
    #         self._reset_statuses()

    def _reset_statuses(self):
        """Helper to reset agent statuses for the diagram."""
        st.session_state.current_batch_statuses = {
            "LlmOrchestrator": "not_started", "MetaIntentTool": "not_started",
            "FrameSamplerTool": "not_started", "FramePrefilterTool": "not_started",
            "FeaturesSelectionTool": "not_started", "SymptomAnalyzerTool": "not_started"
        }

    def _render_sidebar(self):
        """Renders the sidebar UI and returns user inputs."""
        with st.sidebar:
            st.image("assets/abbott.png", width=300)
            st.header("📁 Configuration")
            input_type = st.radio("Select Input Type:", ["CSV File", "Video File"])
            
            if input_type == "CSV File":
                input_file = st.file_uploader("Upload Blendshape CSV", type=['csv'])
                st.session_state.input_type = "csv"
                st.info("💡 CSV should contain blendshapes. AUs will be calculated if not present.")
            else:
                input_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
                st.session_state.input_type = "video"
                st.info("💡 Video will be processed to extract blendshapes and keypoints, then calculate AUs.")
            
            work_dir = st.text_input("Output Directory", value=self.work_dir)
            
            batch_size = st.number_input(
                "Batch Size (Frames per Batch)", 
                min_value=50, max_value=500, value=100, step=50,
                help="Number of rows/frames from the CSV to process in each agent pipeline run."
            )

            user_prompt = st.text_area(
                "Analysis Prompt",
                value="Analyze this data for medical assessment",
                height=100
            )
            
            st.markdown("---")
            process_button = st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True)
            
            if st.button("🔄 Reset", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        return input_file, work_dir, batch_size, user_prompt, process_button

    def _render_header(self):
        """Renders the main title and header."""
        st.title("🏥 Medical AI Agent Pipeline")
        st.markdown("Automated facial analysis using Google ADK agents")
        st.markdown("---")

    def _render_footer(self):
        """Renders the application footer."""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
            <small>Medical AI Agent Pipeline | Powered by Google ADK & Gemini 2.0</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    # def _render_idle_state(self):
    #     """Renders the UI when the app is in its initial, idle state."""
    #     st.plotly_chart(
    #         create_agent_diagram(st.session_state.current_batch_statuses),
    #         use_container_width=True,
    #         key="initial_agent_diagram"
    #     )
    #     st.info("👆 Upload a file, configure batch size, and click 'Start Batch Analysis' to begin.")
    #     with st.expander("📖 How it works"):
    #         st.markdown("""
    #         ### Agent Pipeline Flow
    #         1. **🧭 Orchestrator Agent**: Manages the entire pipeline...
    #         2. **🔍 Frame Prefilter Agent**: Analyzes data...
    #         3. **📊 CSV Filter Agent**: Identifies active facial regions...
    #         4. **🩺 Symptom Analyzer Agent**: Performs clinical analysis...
            
    #         ### Features
    #         - Real-time agent status tracking
    #         - ...
    #         """)

    def _render_results(self):
        """Renders the full results display after processing is complete."""
        st.markdown("---")
        st.header("📊 Batch Analysis Results")

        num_batches = len(st.session_state.batch_data)
        tab_titles = ["📝 Agent Logs"] + [f"Batch {i+1}" for i in range(num_batches)] + ["❓ MetaIntent" , "📈 Final Summary"]
        main_tabs = st.tabs(tab_titles)
        
        # --- Agent Logs Tab ---
        with main_tabs[0]:
            st.subheader("Agent Execution Logs")
            if 'batch_data' in st.session_state and st.session_state.batch_data:
                batch_data = st.session_state.batch_data
                all_logs = []
                for entry in batch_data:
                    all_logs.extend(entry.get('agent_logs', []))
                
                all_agents = sorted(list(set([log['agent'] for log in all_logs])))
                
                col1, col2 = st.columns([3, 1])
                with col1: st.subheader("Global Agent Filter")
                with col2:
                    if st.button("Reset Filters"): st.rerun()
                
                global_filter = st.multiselect(
                    "Show agents across all batches:",
                    all_agents, default=all_agents, key="global_agent_filter"
                )
                st.divider()
                
                grouped_batches = group_logs_by_batch(batch_data)
                
                for batch_id, logs in grouped_batches.items():
                    filtered_logs = [log for log in logs if log['agent'] in global_filter]
                    if filtered_logs:
                        render_batch_section(batch_id, filtered_logs)
                
                with st.expander("📋 Agent Color Legend"):
                    legend_cols = st.columns(3)
                    for idx, (agent, colors) in enumerate(Constants.AGENT_COLORS.items()):
                        with legend_cols[idx % 3]:
                            st.markdown(f"""
                            <div style="background-color: {colors['bg']}; ...">
                                <strong style="color: {colors['text']}">{agent}</strong>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("📭 No batch data available.")

        # --- Loop through each BATCH tab ---
        for i, batch_tab in enumerate(main_tabs[1:-2]):
            with batch_tab:
                batch_info = st.session_state.batch_data[i]
                batch_state_data = batch_info['session_state_data']
                batch_df = batch_info['original_df']

                st.subheader(f"Detailed Analysis for Batch {i+1}")
                st.info(f"This batch contained {len(batch_df)} frames.")

                nested_tab_titles = [
                    "🎭 Blendshapes", "🔢 AUs", "👀 Emotions/Pain",
                    "❓ Sampling", "🔍 Prefilter Results", "📊 Filtered Data",
                    "🩺 Symptom Analysis", "💾 Session State"
                ]
                nested_tabs = st.tabs(nested_tab_titles)

                loaded_data = load_data_from_sources(batch_state_data, batch_df)

                if loaded_data['structure']:
                    with nested_tabs[0]: # Blendshapes
                        if loaded_data['blendshapes_df'] is not None and not loaded_data['blendshapes_df'].empty:
                            st.markdown("### 🎭 Blendshapes by Facial Region")
                            region_tabs = st.tabs(["👁️ Eyes", "👄 Mouth", "👃 Nose"])
                            for r_idx, (region, r_tab) in enumerate(zip(["eyes", "mouth", "nose"], region_tabs)):
                                with r_tab:
                                    render_regional_blendshapes(loaded_data['blendshapes_df'], loaded_data['structure'], region, f"b{i}_blend_{r_idx}")
                        else:
                            st.warning("No blendshape data for this batch.")

                    with nested_tabs[1]: # AUs
                        if loaded_data['aus_df'] is not None and not loaded_data['aus_df'].empty:
                            st.markdown("### 🔢 AUs by Facial Region")
                            au_region_tabs = st.tabs(["👁️ Eyes AUs", "👄 Mouth AUs", "👃 Nose AUs"])
                            for r_idx, (region, r_tab) in enumerate(zip(["eyes", "mouth", "nose"], au_region_tabs)):
                                with r_tab:
                                    render_regional_aus(loaded_data['aus_df'], loaded_data['structure'], region, f"b{i}_au_{r_idx}")
                        else:
                            st.warning("No AU data for this batch.")
                    
                    with nested_tabs[2]: # Emotions/Pain
                        df_to_render = loaded_data['emotions_df'] if loaded_data['emotions_df'] is not None else batch_df
                        render_emotions_pain(df_to_render, loaded_data['structure'], f"b{i}_emo")

                with nested_tabs[3]: # Sampling
                    st.subheader("Sampling Results")
                    sampler_raw = batch_state_data.get("csv_sampler_result", "")
                    st.markdown("#### ⏱️ Frame Sampling Analysis")
                    if sampler_raw and isinstance(sampler_raw, dict):
                        st.metric("Target FPS", sampler_raw.get("target_fps", "N/A"))
                        st.info(f"**LLM Reason:** {sampler_raw.get('reason', 'N/A')}")
                    else:
                        st.warning("Sampling was not performed for this batch.")

                with nested_tabs[4]: # Prefilter
                    st.subheader("Prefilter Agent Results")
                    prefilter_result = batch_state_data.get("prefilter_result")
                    if prefilter_result and isinstance(prefilter_result, dict):
                        st.metric("Useful Frames Found?", "Yes" if prefilter_result.get("useful") else "No")
                        st.info(f"**Identified Frame Ranges:** `{prefilter_result.get('kept_ranges', '[]')}`")
                        st.info(f"**LLM Reason:** {prefilter_result.get('reason', 'N/A')}")
                        st.success(f"Filtered CSV saved to: `{prefilter_result.get('filtered_csv_path')}`")
                    else:
                        st.warning("Prefilter was not run or found no useful frames for this batch.")

                with nested_tabs[5]: # Filtered Data
                    st.subheader("Region Filter Agent Results")
                    filter_result = batch_state_data.get("filtered_csv_result")
                    if filter_result and isinstance(filter_result, dict):
                        st.info(f"**Identified Active Regions:** `{filter_result.get('regions', '[]')}`")
                        st.success(f"Region-filtered CSV saved to: `{filter_result.get('filtered_csv_path')}`")
                    else:
                        st.warning("Region filter was not run for this batch.")

                with nested_tabs[6]: # Symptom Analysis
                    st.subheader(f"Clinical Symptom Analysis for Batch {i+1}")
                    symptom_analysis_raw = batch_state_data.get("symptom_analysis", "")
                    match = re.search(r"```json\s*(\[.*\])\s*```", symptom_analysis_raw, re.DOTALL)
                    if match:
                        try:
                            symptoms_list = json.loads(match.group(1))
                            for idx, symptom in enumerate(symptoms_list):
                                st.success(f"**Symptom:** {symptom.get('symptom', 'N/A')}")
                                st.markdown(f"**Region:** {symptom.get('affected_region', '').capitalize()} | **Confidence:** {symptom.get('confidence', 0):.0%}")
                                with st.expander("Details"):
                                    st.json(symptom)
                        except json.JSONDecodeError:
                            st.error("Could not parse symptom JSON.")
                            st.code(symptom_analysis_raw)
                    else:
                        st.warning("No structured symptom data found for this batch.")
                        if symptom_analysis_raw: st.code(symptom_analysis_raw)
                
                with nested_tabs[7]: # Session State
                    st.subheader(f"Full Session State for Batch {i+1}")
                    st.json(batch_state_data)
        
        # --- MetaIntent Tab ---
        with main_tabs[-2]:
            st.subheader("Meta-Intent")
            # Use data from the last batch, as MetaIntent usually runs first
            if st.session_state.batch_data:
                meta_raw = st.session_state.batch_data[0]['session_state_data'].get("meta_intent_result", "")
                st.markdown("#### 🧩 MetaIntent Analysis")
                if meta_raw:
                    match = re.search(r"(\{.*\})", meta_raw, re.DOTALL)
                    if match:
                        try:
                            intent_data = json.loads(match.group(1))
                            st.metric("Intent Type", intent_data.get("intent_type", "N/A"))
                            st.metric("Disease Focus", intent_data.get("disease_focus") if intent_data.get("disease_focus") != "" else "N/A (General)")
                            st.metric("Sample Data Provided?", "✅ Yes" if intent_data.get("sample_data_provided") else "❌ No")
                            st.info(f"**LLM Reason:** {intent_data.get('reason', 'N/A')}")
                        except json.JSONDecodeError:
                            st.error("Could not parse MetaIntent JSON.")
                            st.code(meta_raw)
                else:
                    st.warning("No MetaIntent data found.")

        # --- Final Summary Tab ---
        with main_tabs[-1]:
            st.header("Overall Clinical Summary Across All Batches")
            if st.session_state.final_summary:
                st.markdown(st.session_state.final_summary)
            else:
                st.warning("The final summary could not be generated or is empty.")

    async def _main_pipeline_async(self):
        """
        The main asynchronous function that orchestrates the batch processing.
        This was converted from the `async def main()` function.
        """
        session_service = InMemorySessionService()
        stop_processing = False

        for i, batch_df in enumerate(self.batches):
            batch_num = i + 1
            self._reset_statuses()
            st.session_state.current_batch_statuses["LlmOrchestrator"] = "running"
            self.status_text.info(f"☑️ Preparing Batch {batch_num} of {self.total_batches}...")
            
            # with self.diagram_placeholder.container():
            #     st.plotly_chart(create_agent_diagram(st.session_state.current_batch_statuses), 
            #                     use_container_width=True, 
            #                     key=f"start_diagram_for_batch_{i}")
            
            await asyncio.sleep(1.5)                 
            self.status_text.info(f"🚀 **Processing Batch {batch_num} of {self.total_batches}...** ({len(batch_df)} frames)")
            
            batch_path = os.path.join(self.work_dir, f"batch_{i}.csv")
            batch_df.to_csv(batch_path, index=False)

            current_batch_logs = []
            event_count_in_batch = 0
            session_id = f"session_batch_{i}_{datetime.now().strftime('%H%M%S')}"
            
            async for event, session_state in run_pipeline_for_batch_async(batch_path, self.work_dir, self.user_prompt, session_service, session_id):
                author = event.author or "System"

                # Update statuses
                if "MetaIntent" in author: 
                    st.session_state.current_batch_statuses["MetaIntentTool"] = "running"
                    await asyncio.sleep(1.5)
                elif "SamplingFrames" in author: st.session_state.current_batch_statuses["FrameSamplerTool"] = "running"
                elif "Prefilter" in author: st.session_state.current_batch_statuses["FramePrefilterTool"] = "running"
                elif "FeaturesSelection" in author or "Region" in author: st.session_state.current_batch_statuses["FeaturesSelectionTool"] = "running"
                elif "Symptom" in author: st.session_state.current_batch_statuses["SymptomAnalyzerTool"] = "running"

                if event.content and event.content.parts:
                    message = " ".join([p.text.strip() for p in event.content.parts if p.text])
                    
                    if "✅" in message:
                        if "MetaIntent analysis" in message:
                            st.session_state.current_batch_statuses["MetaIntentTool"] = "completed"
                            if i == 0:
                                match = re.search(r"(\{.*\})", message, re.DOTALL)
                                if match:
                                    try:
                                        intent_data = ast.literal_eval(match.group(1))
                                        if intent_data.get("intent_type") == "invalid_input":
                                            self._reset_statuses()
                                            error_message = (
                                                "Execution stopped. This application is for facial medical analysis only. "
                                                f"Reason: {intent_data.get('reason', 'The prompt was not related to this tool.')}"
                                            )
                                            self.status_text.error(error_message)
                                            st.warning(error_message)
                                            stop_processing = True
                                            break
                                    except json.JSONDecodeError: pass
                        elif "CSV sampled" in message: st.session_state.current_batch_statuses["FrameSamplerTool"] = "completed"
                        elif "Prefilter kept frames" in message: st.session_state.current_batch_statuses["FramePrefilterTool"] = "completed"
                        elif "Filtered CSV saved" in message: st.session_state.current_batch_statuses["FeaturesSelectionTool"] = "completed"
                        elif "Symptom analysis process completed" in message: st.session_state.current_batch_statuses["SymptomAnalyzerTool"] = "completed"
                        elif "Orchestrator finished flow" in message: st.session_state.current_batch_statuses["LlmOrchestrator"] = "completed"

                    event_count_in_batch += 1
                    # with self.diagram_placeholder.container():
                        # st.plotly_chart(create_agent_diagram(st.session_state.current_batch_statuses), 
                        #                 use_container_width=True, 
                        #                 key=f"diag_b{i}_e{event_count_in_batch}")
                                        
                    self.status_text.info(f"**[Batch {batch_num}]** [{author}] {message[:150]}...")
                    with self.log_container:
                        st.markdown(f'<div class="agent-card {author}"><strong>Batch {batch_num} {author}</strong>: {message}</div>', unsafe_allow_html=True)
                    current_batch_logs.append({"timestamp": datetime.now().strftime("%H:%M:%S"), "agent": author, "message": message})

            if stop_processing:
                self._reset_statuses()
                # with self.diagram_placeholder.container():
                #     st.plotly_chart(create_agent_diagram(st.session_state.current_batch_statuses),
                #                     use_container_width=True,
                #                     key="halt_diagram")
                return 
            
            final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            st.session_state.batch_data.append({
                "batch_id": i, "agent_logs": current_batch_logs,
                "session_state_data": dict(final_session.state),
                "original_df": batch_df 
            })
            self.status_text.success(f"✅ **Batch {batch_num} of {self.total_batches} complete!**")
            self.progress_bar.progress(int((batch_num / self.total_batches) * 100))
            await asyncio.sleep(1) 
            
        self.status_text.info("📊 **Generating final summary across all batches...**")
        summary_input = format_batches_for_summary(st.session_state.batch_data)
        
        async for event in run_summary_agent_async(summary_input, session_service):
            if event.content and event.content.parts:
                st.session_state.final_summary = event.content.parts[0].text

        st.session_state.processing_complete = True
        self.status_text.success("🎉 **All Batches Processed and Summarized!**")
        self._reset_statuses()
        st.rerun()

    def _run_processing_pipeline(self, df_check, batch_size, user_prompt):
        """
        Handles the setup and execution of the asynchronous processing pipeline.
        """
        try:
            os.makedirs(Constants.VISION_OUT_DIR, exist_ok=True)
            
            # if st.session_state.vision_input_type == "video":
            #     pass
            #     # video_path = os.path.join(work_dir, "input_video.mp4")
            #     # with open(video_path, "wb") as f: f.write(input_file.getvalue())
                
            #     # st.info("🎬 Video processing: Extract blendshapes...")
            #     # csv_path = Infer(video_path).inference()
            #     # df_check = pd.read_csv(csv_path)
            # else:
            #     csv_path = os.path.join(Constants.VISION_OUT_DIR, "input_data.csv")
            #     with open(csv_path, "wb") as f: f.write(input_file.getvalue())
            #     df_check = pd.read_csv(csv_path)

            has_aus = any('AU' in col for col in df_check.columns)
            if has_aus: st.info("✅ CSV contains AUs - proceeding.")
            else: st.info("📊 CSV contains only blendshapes - AUs will be calculated.")
            
            st.session_state.original_blendshapes_df = df_check
            
            # Set instance attributes for the async method
            self.batches = create_batches(df_check, batch_size)
            self.total_batches = len(self.batches)
            self.work_dir = Constants.VISION_OUT_DIR
            self.user_prompt = user_prompt
            
            # Create UI placeholders and store them as instance attributes
            self.diagram_placeholder = st.empty()
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.log_container = st.container()

            st.session_state.batch_data = [] # Clear previous results
            
            # Run the main async pipeline
            asyncio.run(self._main_pipeline_async())

        except Exception as e:
            st.error(f"An error occurred during the pipeline execution: {e}")
            logger.error("Pipeline failed", exc_info=True)

    def run(self):
        """
        The main public method to run the Streamlit application.
        This method controls the overall UI flow.
        """
        self._render_header()
        
        # Get user inputs from the sidebar
        input_file, work_dir, batch_size, user_prompt, process_button = self._render_sidebar()
        
        # Main application logic
        if process_button and input_file:
            st.session_state.processing_complete = False # Start processing
            self._run_processing_pipeline(input_file, work_dir, batch_size, user_prompt)
        
        elif st.session_state.processing_complete:
            # If processing is done, show results
            self._render_results()
            
        else:
            # Otherwise, show the idle state
            self._render_idle_state()
            
        self._render_footer()