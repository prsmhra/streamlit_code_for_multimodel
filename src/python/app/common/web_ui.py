
import re
import cv2
import tempfile
import asyncio
import os
import os
from pydub import AudioSegment
import math
import streamlit as st
import json
import librosa
import time
import hashlib
from pathlib import Path

 
from src.python.app.constants.constants import Constants
from src.python.app.utils.show_input_file import *
from src.python.app.utils.agents_logs import *
from src.python.app.utils.data_utils import *
from src.python.app.utils.thread_pool_executer import *
from src.python.app.utils.ui_renders import *

from src.python.app.common.vision_agent_call import MedicalAIAgentApp
from src.python.app.common.audio_agent_integration import AudioAgentPipeline
from src.python.app.video_frame_extractor.csv_sav_inference import Infer 

# Import multimodal components
from src.python.app.multimodal.run_mutlimodal import run_full_pipeline
from src.python.app.multimodal.video_utils import extract_audio_from_video


@st.cache_data
def load_csv(path_str):
    if not path_str or not Path(path_str).exists():
        return None
    try:
        return pd.read_csv(path_str)
    except Exception as e:
        st.error(f"Error loading {path_str}: {e}")
        return None

# --- Helper function to safely load a JSON file ---
@st.cache_data
def load_json(path_str):
    if not path_str or not Path(path_str).exists():
        return None
    try:
        with open(path_str, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {path_str}: {e}")
        return None

# --- Helper function to parse the RF guidance string ---
@st.cache_data
def parse_rf_guidance(guidance_str):
    if not guidance_str:
        return None
    try:
        # Extracts the JSON list part from the string
        json_part = guidance_str.split(':', 1)[1].strip()
        return pd.DataFrame(json.loads(json_part))
    except Exception as e:
        st.error(f"Error parsing RF guidance: {e}")
        return None

 

 
class webUI:
    def __init__(self):
        """
            This is __init__ method contains global variables of class
        """
        # DO NOT set page config here - it must be done in inference.py BEFORE this is called
        self.ui_style()
 
        # Initialize session state - ALIGNED WITH MedicalAIAgentApp
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = []
        if 'final_summary' not in st.session_state:
            st.session_state.final_summary = ""
        if 'current_batch_statuses' not in st.session_state:
            st.session_state.current_batch_statuses = {}
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'currently_processing' not in st.session_state:
            st.session_state.currently_processing = False
        if 'current_batch_index' not in st.session_state:
            st.session_state.current_batch_index = 0
        if 'expected_total_batches' not in st.session_state:
            st.session_state.expected_total_batches = 0
        if Constants.SELECTED_MODE not in st.session_state:
            st.session_state.selected_mode = None
        if Constants.UPLOADED_FILE_KEY not in st.session_state:
            st.session_state.uploaded_files_key = 0
        if Constants.VISION_INPUT_TYPE not in st.session_state:
            st.session_state.vision_input_type = None
        if Constants.MULTIMODAL_VIDEO_TYPE not in st.session_state:
            st.session_state.multi_video_type = None
        
        # Audio-specific session state
        if 'audio_chunk_results' not in st.session_state:
            st.session_state.audio_chunk_results = []
        if 'audio_processing' not in st.session_state:
            st.session_state.audio_processing = False
        if 'audio_batch_data' not in st.session_state:
            st.session_state.audio_batch_data = []
        
        # Audio-specific session state
        if 'audio_chunk_results' not in st.session_state:
            st.session_state.audio_chunk_results = []
        if 'audio_processing' not in st.session_state:
            st.session_state.audio_processing = False
        if 'audio_batch_data' not in st.session_state:
            st.session_state.audio_batch_data = []

        #Multimodel specific session state
        if 'multimodal_results' not in st.session_state:
            st.session_state.multimodal_results = {}
        if 'multimodal_processing' not in st.session_state:
            st.session_state.multimodal_processing = False
        if 'multimodal_batch_data' not in st.session_state:
            st.session_state.multimodal_batch_data = []
 
        ## class variables
        self.user_prompt = None
        self.uploaded_file = None
        self.audio_file = None
        self.input_file_show_container = None
        self.process_button = None
        self.vision_batch_size = None
        self.work_dir = Constants.WORK_DIR  # NEW: Work directory
        self.video_fps = Constants.DEFAULT_FPS
        self.video_duration = None
        self.audio_duration = None
        self.total_batches = Constants.ONE
        self.frame_count = None
        self.df = None  # Store DataFrame for preview
        self.audio_file_name = None
        self.audio_overlap = None
        self.audio_top_features = None
        self.multi_overlap_size = None
        self.multimodal_batch_size = None
        self.multi_top_features = None
 
        # Single instance of MedicalAIAgentApp
        self.vision_medical_agent = MedicalAIAgentApp()
        self.audio_agent = AudioAgentPipeline()
        self.audio_agent = AudioAgentPipeline()
 
 
    def ui_style(self):
        """
            This method contains the style of the web page
        """
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #6a0dad;
                text-align: center;
                margin-bottom: 1rem;
                background: linear-gradient(90deg, #6a0dad, #1e90ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .stage-box {
                background-color: #f8f7ff;
                padding: 1rem;
                border-radius: 10px;
                border-left: 5px solid #6a0dad;
                margin: 1rem 0;
            }
            .success-box {
                background-color: #e6f7f0;
                padding: 1rem;
                border-radius: 5px;
                border-left: 5px solid #00b894;
            }
            .processing-box {
                background-color: #fffaf0;
                padding: 1rem;
                border-radius: 5px;
                border-left: 5px solid #fdcb6e;
            }
            .metric-card {
                background-color: white;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(106, 13, 173, 0.1);
                border: 1px solid #e6e6fa;
            }
            .input-section {
                background-color: #fafaff;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border: 1px solid #e6e6fa;
            }
            .file-input-section {
                background-color: #f0f8ff;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border: 1px dashed #6a0dad;
            }
            .stButton button {
                background-color: #6a0dad;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0.5rem 1rem;
            }
            .stButton button:hover {
                background-color: #5a0c9a;
                color: white;
            }
            .mode-selector {
                background: linear-gradient(135deg, #6a0dad, #1e90ff);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin-bottom: 1rem;
            }
            
            /* Enhanced Agent Log Styles */
            .agent-log-card {
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 8px;
                border-left: 4px solid;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                animation: slideIn 0.3s ease-out;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .agent-log-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.5rem;
            }
            
            .agent-name {
                font-weight: bold;
                font-size: 1.1em;
            }
            
            .agent-timestamp {
                font-size: 0.85em;
                opacity: 0.7;
                color: black;
            }
            
            .agent-message {
                font-size: 0.95em;
                line-height: 1.5;
                margin-top: 0.5rem;
            }
            
            .batch-badge {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                margin-right: 0.5rem;
            }
            
            /* Agent-specific colors with enhanced contrast */
            .MetaIntentLLM, .MetaIntentTool {
                background: linear-gradient(135deg, #FFE5E5 0%, #FFD1D1 100%);
                border-color: #FF6B6B;
            }
            .MetaIntentLLM .agent-name, .MetaIntentTool .agent-name {
                color: #8B0000;
            }
            
            .PipelineOrchestratorLLM, .LlmOrchestrator {
                background: linear-gradient(135deg, #E5F3FF 0%, #D1E7FF 100%);
                border-color: #4DABF7;
            }
            .PipelineOrchestratorLLM .agent-name, .LlmOrchestrator .agent-name {
                color: #0B3BA1;
            }
            
            .FrameSamplerTool, .SamplingFramesLLM {
                background: linear-gradient(135deg, #E5FCFF 0%, #D1F5FF 100%);
                border-color: #74C0FC;
            }
            .FrameSamplerTool .agent-name, .SamplingFramesLLM .agent-name {
                color: #004E89;
            }
            
            .FramePrefilterTool, .PrefilterLLM {
                background: linear-gradient(135deg, #F0E5FF 0%, #E5D1FF 100%);
                border-color: #B197FC;
            }
            .FramePrefilterTool .agent-name, .PrefilterLLM .agent-name {
                color: #5F00B2;
            }
            
            .FeaturesSelectionTool, .RegionDetectorLLM {
                background: linear-gradient(135deg, #FFF4E5 0%, #FFE8D1 100%);
                border-color: #FFD43B;
            }
            .FeaturesSelectionTool .agent-name, .RegionDetectorLLM .agent-name {
                color: #995A00;
            }
            
            .SymptomAnalyzerTool, .SymptomReasonerLLM {
                background: linear-gradient(135deg, #E5F5F0 0%, #D1F0E5 100%);
                border-color: #51CF66;
            }
            .SymptomAnalyzerTool .agent-name, .SymptomReasonerLLM .agent-name {
                color: #0B5F0B;
            }
            
            /* Processing animation */
            .processing-log {
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                    transform: scale(1);
                }
                50% {
                    opacity: 0.9;
                    transform: scale(1.02);
                }
            }
            
            /* Success indicator */
            .success-indicator {
                color: #00b894;
                font-weight: bold;
            }
            
            /* Error indicator */
            .error-indicator {
                color: #d63031;
                font-weight: bold;
            }
            
            /* Progress tracker styles */
            .progress-tracker {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 1rem 0;
            }
            
            .batch-progress-container {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin: 1rem 0;
            }
            
            .batch-status-icon {
                font-size: 2em;
            }
            
            .batch-status-pending {
                color: #95a5a6;
            }
            
            .batch-status-processing {
                color: #f39c12;
                animation: rotate 2s linear infinite;
            }
            
            .batch-status-complete {
                color: #27ae60;
            }
            
            .batch-status-error {
                color: #e74c3c;
            }
            
            @keyframes rotate {
                from {
                    transform: rotate(0deg);
                }
                to {
                    transform: rotate(360deg);
                }
            }
            
            /* Timeline style for batch flow */
            .batch-timeline {
                position: relative;
                padding-left: 2rem;
            }
            
            .batch-timeline-item {
                position: relative;
                padding-bottom: 2rem;
            }
            
            .batch-timeline-item:before {
                content: '';
                position: absolute;
                left: -1.5rem;
                top: 0;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                border: 2px solid #6a0dad;
            }
            
            .batch-timeline-item.completed:before {
                background: #27ae60;
                border-color: #27ae60;
            }
            
            .batch-timeline-item.processing:before {
                background: #f39c12;
                border-color: #f39c12;
                animation: pulse 1s infinite;
            }
            
            .batch-timeline-item:after {
                content: '';
                position: absolute;
                left: -1.45rem;
                top: 12px;
                width: 2px;
                height: calc(100% - 12px);
                background: #e0e0e0;
            }
            
            .batch-timeline-item:last-child:after {
                display: none;
            }
        </style>
        """, unsafe_allow_html=True)
 
    def input_file_upload(self):
        """
            This method contains the file upload and mode selection code
        """
        # Mode Selection Section
        col_mode, col_space = st.columns(Constants.TWO)
        with col_mode:
            processing_mode = st.selectbox(
                Constants.MODEL_SELECTION_STR,
                Constants.SELECTION_LIST,
                help=Constants.SELECTION_HELP_STR,
                key=Constants.SELECTION_KEY
            )
 
            if processing_mode == Constants.VISION_STR:
                vision_input_typ = st.radio(
                        Constants.INPUT_TYPE_STR,
                        Constants.VISION_INPUT_TYPE_LIST,
                        key=Constants.VISION_TYPE_STR
                    )
                st.session_state.vision_input_type = vision_input_typ

            if processing_mode == Constants.MULTIMODAL_STR:
                multi_video_typ = st.radio(
                        Constants.INPUT_TYPE_STR,
                        Constants.MULTI_INPUT_TYPE_LIST,
                        key=Constants.VISION_TYPE_STR
                    )
                st.session_state.multi_video_type = multi_video_typ
        
        # Dynamic file upload based on mode
        with col_space:
            # Video
            if processing_mode == Constants.VISION_STR:
                st.markdown(Constants.VISION_HEADING)
                st.session_state.selected_mode = Constants.VISION_STR
                
                if vision_input_typ == Constants.VIDEO_FILES_STR:
                    self.uploaded_file = st.file_uploader(
                        Constants.VIDEO_UPLOAD_STR,
                        type=Constants.VIDEO_EXT,
                        accept_multiple_files=False,
                        key=f"{Constants.VIDEO_UPLOAD_KEY}{Constants.UNDERSCORE}{st.session_state.uploaded_files_key}"
                    )
                    if self.uploaded_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Constants.MP4) as tmp_file:
                            tmp_file.write(self.uploaded_file.read())
                            tmp_path = tmp_file.name
 
                        # Use OpenCV to read video and get FPS
                        video = cv2.VideoCapture(tmp_path)
                        self.video_fps = video.get(cv2.CAP_PROP_FPS) if video.get(cv2.CAP_PROP_FPS) else Constants.DEFAULT_FPS
                        self.frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                        self.video_duration = int(self.frame_count / self.video_fps if self.video_fps > Constants.ZERO else Constants.DEFAULT_FPS)
                        
                        video.release()
                        # Reset file pointer for later use
                        self.uploaded_file.seek(Constants.ZERO)
 
                else:
                    self.uploaded_file = st.file_uploader(
                        Constants.CSV_FILES_STR,
                        type=Constants.CSV_EXT,
                        accept_multiple_files=False,
                        key=f"{Constants.CSV_UPLOAD_KEY}_{st.session_state.uploaded_files_key}"
                    )
                    
            # Audio
            elif processing_mode == Constants.AUDIO_STR:
                st.markdown(Constants.AUDIO_HEADING)
                st.session_state.selected_mode = Constants.AUDIO_STR
                self.uploaded_file = st.file_uploader(
                    Constants.AUDIO_UPLOAD_STR,
                    type=Constants.AUDIO_EXT,
                    accept_multiple_files=False,
                    key=f"{Constants.AUDIO_UPLOAD_KEY}_{st.session_state.uploaded_files_key}"
                )
                if self.uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                        tmp_audio.write(self.uploaded_file.read())
                        audio_path = tmp_audio.name
                    try:
                        y, sr = librosa.load(audio_path, sr=None)
                        self.audio_duration = math.ceil(librosa.get_duration(y=y, sr=sr))
                    except Exception:
                        audio = AudioSegment.from_file(audio_path)
                        self.audio_duration = math.ceil(len(audio) / float(Constants.THOUSAND))
                    
                    self.uploaded_file.seek(Constants.ZERO)
                    self.audio_file_name = self.uploaded_file.name
 
            # Multimodal
            else:  
                st.markdown(Constants.MULTIMODEL_HEADING)
                st.session_state.selected_mode = Constants.MULTIMODAL_STR
                if st.session_state.multi_video_type == Constants.VIDEO_FILES_STR:
                    self.uploaded_file = st.file_uploader(
                        Constants.VIDEO_UPLOAD_STR,
                        type=Constants.VIDEO_EXT,
                        accept_multiple_files=False,
                        key=f"{Constants.MULTIMODEL_VIDEO_KEY}_{st.session_state.uploaded_files_key}"
                    )
                elif st.session_state.multi_video_type == Constants.VISION_CSV_AND_AUDIO:
                    csv, audio = st.columns(Constants.TWO)
                    with csv:
                        self.uploaded_file = st.file_uploader(
                            Constants.CSV_FILES_STR,
                            type=Constants.CSV_EXT,
                            accept_multiple_files=False,
                            key=f"multi_csv_{st.session_state.uploaded_files_key}"
                        )
                    with audio: 
                        self.audio_file = st.file_uploader(
                            Constants.AUDIO_UPLOAD_STR,
                            type=Constants.AUDIO_EXT,
                            accept_multiple_files=False,
                            key=f"multi_audio_{st.session_state.uploaded_files_key}"
                        )
 
            st.markdown('</div>', unsafe_allow_html=True)
        
        self.show_input_file()
        self.user_input_prompt()
        process_col, reset_col = st.columns(Constants.TWO)
        st.markdown(
                """
                <style>
                div.stButton > button:hover {
                    cursor: pointer; /* Change to a pointer cursor on hover */
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        with process_col:
            self.process_button = st.button(Constants.START_ANAYSIS_KEY, type=Constants.BUTTON_TYPE_KEY, use_container_width=True)
        with reset_col:
            if st.button(Constants.RESET_BTN_KEY, use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        st.divider()
 
    def show_input_file(self):
        if self.uploaded_file or self.audio_file:
            with st.expander(Constants.INPUT_SHOW_EXPAND):
                self.input_file_show_container = st.empty()
                audio_container = st.empty()
                
                file_name = self.uploaded_file.name
                file_ext = file_name.split(Constants.DOT)[-Constants.ONE].lower()
                
                if  st.session_state.selected_mode in [Constants.VISION_STR, Constants.MULTIMODAL_STR] and file_ext in Constants.CSV_EXT:
                    self.df = show_csv_files(file_name, self.uploaded_file, self.input_file_show_container)
                    self.video_fps = Constants.DEFAULT_FPS
                    self.video_duration = int(self.df[Constants.FRAME_KEY].nunique() / self.video_fps)
                    self.frame_count = self.df[Constants.FRAME_KEY].nunique()
                    # Reset file pointer
                    self.uploaded_file.seek(Constants.ZERO)
                    
                elif st.session_state.selected_mode == Constants.AUDIO_STR:
                    show_audio_waveform(self.audio_file, self.input_file_show_container, audio_container)
                    
                elif st.session_state.selected_mode in [Constants.VISION_STR, Constants.MULTIMODAL_STR] and file_ext in Constants.VIDEO_EXT:
                    # self.input_file_show_container.video(self.uploaded_file)
                    # Reset file pointer
                    show_video_preview(self.uploaded_file, self.input_file_show_container)
                    self.uploaded_file.seek(Constants.ZERO)

                if self.audio_file:
                    st.divider()
                    show_audio_waveform(self.audio_file, self.input_file_show_container, audio_container)
        
    def vision_ui(self):
        """
            This method implements vision results
        """
        # Check if we have batch data
        num_batches = len(st.session_state.batch_data)
        if num_batches == Constants.ZERO and st.session_state.processing_complete:
            st.warning("No batch data available yet. Processing may have failed or not started.")
            return
        
        batch_tabs = st.tabs([Constants.AGENT_LOGS_HEADING] + 
                             [f"{Constants.BATCH_KEY} {i+Constants.ONE}" for i in range(num_batches)] + 
                             ["⚕️ MetaIntent", Constants.FINAL_SUMMERY_KEY])
        
        with batch_tabs[Constants.ZERO]:
            st.subheader(Constants.AGENT_LOGS_HEADING)
            if len(st.session_state.batch_data) > 0:
                batch_data = st.session_state.batch_data
                all_logs = []
                for entry in batch_data:
                    all_logs.extend(entry.get(Constants.AGENTS_LOGS_KEY, []))
                
                all_agents = sorted(list(set([log[Constants.AGENT_KEY] for log in all_logs])))
                
                col1, col2 = st.columns([Constants.THREE, Constants.ONE])
                with col1:
                    st.subheader(Constants.GLOBAL_AGENT_FILTER_KEY)
                with col2:
                    if st.button(Constants.RESET_FILTER_KEY):
                        st.rerun()
                
                global_filter = st.multiselect(
                    Constants.MULTISELECT_KEY,
                    all_agents, default=all_agents,
                    key=Constants.UNDERSCORE.join(Constants.GLOBAL_AGENT_FILTER_KEY.split())
                )
                st.divider()
                
                # Enhanced log display with styling
                for batch_idx, batch_entry in enumerate(batch_data):
                    batch_logs = batch_entry.get(Constants.AGENTS_LOGS_KEY, [])
                    filtered_logs = [log for log in batch_logs if log[Constants.AGENT_KEY] in global_filter]
                    
                    if filtered_logs:
                        st.markdown(f"### 📦 Batch {batch_idx + Constants.ONE}")
                        for log in filtered_logs:
                            agent_name = log[Constants.AGENT_KEY]
                            message = log.get('message_key', '')
                            timestamp = log.get('timestamp', '')
                            # Determine if success or error
                            is_success = '✅' in message or 'complete' in message.lower() or 'success' in message.lower()
                            is_error = '❌' in message or 'error' in message.lower() or 'failed' in message.lower()
                            
                            # Create enhanced log card
                            log_html = f"""
                            <div class="agent-log-card {agent_name}">
                                <div class="agent-log-header">
                                    <div>
                                        <span class="batch-badge" style="background: rgba(106, 13, 173, 0.1); color: #6a0dad;"> 
                                            Batch {batch_idx + Constants.ONE} 
                                        </span> 
                                        <span class="agent-name">{agent_name}</span> 
                                    </div> 
                                    <div class="agent-timestamp">{timestamp}</div> 
                                </div> 
                                <div class="agent-message"> 
                                    {message if len(message) < 300 else message[:297] + '...'} 
                                </div> 
                            </div> 
                            """
                            st.write(f"{message if len(message) < 300 else message[:297] + '...'}")
                            
                            st.markdown(log_html, unsafe_allow_html=True)
                        
                        st.divider()
                
                with st.expander(f"📋 {Constants.AGENT_EXPENDER_KEY}"):
                    legend_cols = st.columns(Constants.THREE)
                    
                    for idx, (agent, colors) in enumerate(Constants.AGENT_COLORS.items()):
                        with legend_cols[idx % Constants.THREE]:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {colors['bg']} 0%, {colors['bg']}dd 100%); 
                                        padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0;
                                        border-left: 4px solid {colors['border']}; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <strong style="color: {colors[Constants.TEXT_KEY]};">{agent}</strong>
                            </div>
                            """, unsafe_allow_html=True)
            # elif len(st.session_state.batch_data) == 0:
            #     st.info(f"🔭 {Constants.NO_BATCH_KEY}")
 
        # Batch detail tabs
        if len(st.session_state.batch_data) >= Constants.ONE:
            for i, batch_tab in enumerate(batch_tabs[Constants.ONE:-2]):
                with batch_tab:
                    batch_info = st.session_state.batch_data[i]
                    batch_state_data = batch_info['session_state_data']
                    batch_df = batch_info['original_df']
                    st.subheader(f"Detailed Analysis for Batch {i+Constants.ONE}")
                    st.info(f"This batch contained {len(batch_df)} frames.")
 
                    nested_tab_titles = [
                        "🎭 Blendshapes", "🔢 AUs", "👀 Emotions/Pain",
                        "⏱️ Sampling", "🔍 Prefilter Results", "📊 Filtered Data",
                        "🩺 Symptom Analysis", "💾 Session State"
                    ]
                    nested_tabs = st.tabs(nested_tab_titles)
 
                    loaded_data = load_data_from_sources(batch_state_data, batch_df)
                    # Safety check
                    if not loaded_data or not loaded_data.get('structure'):
                        st.error(f"Could not load data for batch {i+Constants.ONE}. The CSV structure may be incompatible.")
                        st.info("Expected columns: blendshapes (eyeBlink*, mouth*, etc.) or AUs (AU01, AU02, etc.)")
                        with st.expander("Show available columns"):
                            st.write(batch_df.columns.tolist())
                        continue
 
                    if loaded_data['structure']:
                        with nested_tabs[Constants.ZERO]:  # Blendshapes
                            if loaded_data['blendshapes_df'] is not None and not loaded_data['blendshapes_df'].empty:
                                st.markdown("### 🎭 Blendshapes by Facial Region")
                                region_tabs = st.tabs(["👁️ Eyes", "👄 Mouth", "👃 Nose"])
                                for r_idx, (region, r_tab) in enumerate(zip(["eyes", "mouth", "nose"], region_tabs)):
                                    with r_tab:
                                        render_regional_blendshapes(loaded_data['blendshapes_df'], loaded_data['structure'], region, f"b{i}_blend_{r_idx}")
                            else:
                                st.warning("No blendshape data for this batch.")
 
                        with nested_tabs[Constants.ONE]:  # AUs
                            if loaded_data['aus_df'] is not None and not loaded_data['aus_df'].empty:
                                st.markdown("### 🔢 AUs by Facial Region")
                                au_region_tabs = st.tabs(["👁️ Eyes AUs", "👄 Mouth AUs", "👃 Nose AUs"])
                                for r_idx, (region, r_tab) in enumerate(zip(["eyes", "mouth", "nose"], au_region_tabs)):
                                    with r_tab:
                                        render_regional_aus(loaded_data['aus_df'], loaded_data['structure'], region, f"b{i}_au_{r_idx}")
                            else:
                                st.warning("No AU data for this batch.")
                        
                        with nested_tabs[Constants.TWO]:  # Emotions/Pain
                            df_to_render = loaded_data['emotions_df'] if loaded_data['emotions_df'] is not None else batch_df
                            render_emotions_pain(df_to_render, loaded_data['structure'], f"b{i}_emo")
 
                    with nested_tabs[Constants.THREE]:  # Sampling
                        st.subheader("Sampling Results")
                        sampler_raw = batch_state_data.get("csv_sampler_result", "")
                        st.markdown("#### ⏱️ Frame Sampling Analysis")
                        if sampler_raw and isinstance(sampler_raw, dict):
                            st.metric("Target FPS", sampler_raw.get("target_fps", "N/A"))
                            st.info(f"**LLM Reason:** {sampler_raw.get('reason', 'N/A')}")
                        else:
                            st.warning("Sampling was not performed for this batch.")
 
                    with nested_tabs[Constants.FOUR]:  # Prefilter
                        st.subheader("Prefilter Agent Results")
                        prefilter_result = batch_state_data.get("prefilter_result")
                        if prefilter_result and isinstance(prefilter_result, dict):
                            st.metric("Useful Frames Found?", "Yes" if prefilter_result.get("useful") else "No")
                            st.info(f"**Identified Frame Ranges:** `{prefilter_result.get('kept_ranges', '[]')}`")
                            st.info(f"**LLM Reason:** {prefilter_result.get('reason', 'N/A')}")
                            st.success(f"Filtered CSV saved to: `{prefilter_result.get('filtered_csv_path')}`")
                        else:
                            st.warning("Prefilter was not run or found no useful frames for this batch.")
 
                    with nested_tabs[Constants.FIVE]:  # Filtered Data
                        st.subheader("Region Filter Agent Results")
                        filter_result = batch_state_data.get("filtered_csv_result")
                        if filter_result and isinstance(filter_result, dict):
                            st.info(f"**Identified Active Regions:** `{filter_result.get('regions', '[]')}`")
                            st.success(f"Region-filtered CSV saved to: `{filter_result.get('filtered_csv_path')}`")
                        else:
                            st.warning("Region filter was not run for this batch.")
 
                    with nested_tabs[Constants.SIX]:  # Symptom Analysis
                        st.subheader(f"Clinical Symptom Analysis for Batch {i+Constants.ONE}")
                        symptom_analysis_raw = batch_state_data.get("symptom_analysis", "")
                        match = re.search(r"```json\s*(\[.*\])\s*```", symptom_analysis_raw, re.DOTALL)
                        if match:
                            try:
                                symptoms_list = json.loads(match.group(Constants.ONE))
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
                            if symptom_analysis_raw:
                                st.code(symptom_analysis_raw)
                    
                    with nested_tabs[Constants.SEVEN]:  # Session State
                        st.subheader(f"Full Session State for Batch {i+Constants.ONE}")
                        st.json(batch_state_data)
                        pretty = json.dumps(batch_state_data, indent=Constants.TWO, ensure_ascii=False)
                        digest = hashlib.md5(pretty.encode("utf-8")).hexdigest()[:Constants.TEN]
                        dl_key = f"{Constants.VISION_STR.lower()}{Constants.UNDERSCORE}{idx}{Constants.UNDERSCORE}{digest}"
                        st.download_button("⬇️ Download JSON", pretty,
                            file_name="gemini_vision_analysis.json",
                            mime="application/json",
                            key = dl_key,
                            use_container_width=True)
        
        # MetaIntent Tab
        with batch_tabs[-Constants.TWO]:
            st.subheader("Meta-Intent Analysis")
            if st.session_state.batch_data:
                meta_raw = st.session_state.batch_data[0]['session_state_data'].get("meta_intent_result", "")
                st.markdown("#### 🧩 MetaIntent Classification")
                if meta_raw:
                    match = re.search(r"(\{.*\})", meta_raw, re.DOTALL)
                    if match:
                        try:
                            intent_data = json.loads(match.group(Constants.ONE))
                            st.metric("Intent Type", intent_data.get("intent_type", "N/A"))
                            st.metric("Disease Focus", intent_data.get("disease_focus") if intent_data.get("disease_focus") != "" else "N/A (General)")
                            st.metric("Sample Data Provided?", "✅ Yes" if intent_data.get("sample_data_provided") else "❌ No")
                            st.info(f"**LLM Reason:** {intent_data.get('reason', 'N/A')}")
                        except json.JSONDecodeError:
                            st.error("Could not parse MetaIntent JSON.")
                            st.code(meta_raw)
                else:
                    st.warning("No MetaIntent data found.")
        
        # Final Summary Tab
        with batch_tabs[-Constants.ONE]:
            st.header("Overall Clinical Summary Across All Batches")
            if st.session_state.final_summary:
                st.markdown(st.session_state.final_summary)
            else:
                st.warning("The final summary could not be generated or is empty.")   

    def audio_ui(self):
        """
            This method implements audio processing results
        """
        num_batches = len(st.session_state.audio_batch_data)
        if num_batches == Constants.ZERO and st.session_state.processing_complete:
            st.warning("No audio batch data available yet. Processing may have failed or not started.")
            return
        
        batch_tabs = st.tabs([Constants.AGENT_LOGS_HEADING] + 
                             [f"Audio Batch {i+Constants.ONE}" for i in range(num_batches)] + 
                             [Constants.FINAL_SUMMERY_KEY])
        
        with batch_tabs[Constants.ZERO]:
            st.subheader("Audio Processing Logs")
            if len(st.session_state.audio_batch_data) > Constants.ZERO:
                for i, batch in enumerate(st.session_state.audio_batch_data):
                    with st.expander(f"Audio Batch {i+Constants.ONE} - \
                                     {batch.get('start_s', Constants.ZERO):.2f}s to \
                                     {batch.get('start_s', Constants.ZERO) + \
                                      batch.get('duration_s', Constants.ZERO):.2f}s"):
                        if batch.get('error'):
                            st.error(f"Error: {batch['error']}")
                        else:
                            st.success(f"Processed successfully")
                            st.markdown(f"**Batch Directory:** `{batch.get('batch_dir', 'N/A')}`")
            else:
                st.info("No audio processing logs available")
        
        # Batch detail tabs
        for i, batch_tab in enumerate(batch_tabs[Constants.ONE:-Constants.ONE]):
            with batch_tab:
                batch_info = st.session_state.audio_batch_data[i]
                
                st.subheader(f"Audio Batch {i+Constants.ONE} Analysis")
                st.info(f"Time range: {batch_info.get('start_s', Constants.ZERO):.2f}s - \
                        {batch_info.get('start_s', Constants.ZERO) + \
                         batch_info.get('duration_s', Constants.ZERO):.2f}s")
                
                if batch_info.get('error'):
                    st.error(f"Processing failed: {batch_info['error']}")
                else:
                    result_text = batch_info.get('result', 'No result available')
                    
                    
                    render_audio_json_result(result_text, i)
                    
                    # Show batch directory
                    if batch_info.get('batch_dir'):
                        st.markdown(f"**Output Directory:** `{batch_info['batch_dir']}`")
        
        # Final Summary Tab
        with batch_tabs[-Constants.ONE]:
            st.header("Overall Audio Analysis Summary")
            if st.session_state.final_summary:
                st.markdown(st.session_state.final_summary)
            else:
                st.warning("The final audio summary could not be generated or is empty.")
        
    def multimodel_ui(self):
        """
            This method implements multimodal processing results
        """
        if not st.session_state.multimodal_results and not st.session_state.processing_complete:
            st.info("No multimodal results available yet. Start processing to see results.")
            return
        
        # Create tabs for different aspects of multimodal results
        multi_tabs = ["📊 Overview"]+[f"Batch {i}" for i in range(len(st.session_state.multimodal_batch_data))]+["📝 Final Summary"]
        tabs = st.tabs(multi_tabs)
        
        with tabs[0]:  # Overview
            st.subheader("Multimodal Processing Overview")
            
            if st.session_state.multimodal_results:
                col1, col2, col3 = st.columns(Constants.THREE)
                
                with col1:
                    st.metric("Video Path", "✅ Processed")
                    if 'video_path' in st.session_state.multimodal_results:
                        st.caption(st.session_state.multimodal_results['video_path'])
                
                with col2:
                    st.metric("Audio Extracted", "✅ Success" if 'audio_path' in st.session_state.multimodal_results else "❌ Failed")
                    if 'audio_path' in st.session_state.multimodal_results:
                        st.caption(st.session_state.multimodal_results['audio_path'])
                
                with col3:
                    st.metric("Vision CSV", "✅ Generated" if 'csv_path' in st.session_state.multimodal_results else "❌ Failed")
                    if 'csv_path' in st.session_state.multimodal_results:
                        st.caption(st.session_state.multimodal_results['csv_path'])
                
                st.divider()
                
                # Show batch information
                num_batches = len(st.session_state.multimodal_batch_data)
                st.metric("Total Batches Processed", num_batches)
            
                if st.session_state.multimodal_batch_data:
                    st.subheader("Batch Processing Summary")
                    
                    # Get the batch dictionary
                    batches_dict = st.session_state.multimodal_batch_data
                    
                    # Parse the dictionary into a list for the DataFrame
                    summary_data = []
                    for batch_name, batch_data in batches_dict.items():
                        status = batch_data.get('status', 'Unknown')
                        
                        # Set a clear details message
                        details = "See batch tab for details."
                        if status == "skipped":
                            details = f"Skipped: {batch_data.get('validation_error', 'Unknown reason')}"
                        elif status.startswith("processed"):
                            details = "✅ Success"
                        
                        summary_data.append({
                            "Batch Name": batch_name,
                            "Status": status,
                            "Details": details
                        })

                    # Create and display the DataFrame
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                    else:
                        st.info("No batch data available")
    
        batches_dict = st.session_state.get("multimodal_batch_data", {})

        
        batch_items = list(batches_dict.items())

        for i, batch_tab in enumerate(tabs[Constants.ONE:-Constants.ONE]):
            with batch_tab:
                st.subheader(f"🎯 Multimodal Batch {i + Constants.ONE} Results")

                # Check if data for this batch index exists
                if (
                    not batches_dict
                    or i >= len(batch_items)
                ):
                    st.info("No batch data available for this segment. Run the pipeline first.")
                    continue  # Go to the next tab

                # Get the correct batch's name and data using the index
                batch_name, batch_data = batch_items[i]

                # --- Content from your reference code, now inside the tab ---
                
                status = batch_data.get('status', 'unknown')
                st.caption(f"Batch Name: **{batch_name}** |  Status: **{status}**")
                st.divider()

                # If the batch was skipped, just show the error
                if status == "skipped":
                    st.warning(f"This batch was skipped. Reason: {batch_data.get('validation_error', 'Unknown')}")
                    continue  # Go to the next tab

                # --- 1. Display Inputs ---
                st.markdown("#### 1. Batch Inputs")
                col1, col2 = st.columns(Constants.TWO)
                
                with col1:
                    st.markdown("**Input Audio Chunk**")
                    audio_path = batch_data.get('input_audio_path')
                    if audio_path and Path(audio_path).exists():
                        st.audio(audio_path)
                    else:
                        st.error(f"Input audio not found: {audio_path}")
                        
                with col2:
                    st.markdown("**Input Vision CSV (Preview)**")
                    csv_path = batch_data.get('input_vision_csv_path')
                    input_df = load_csv(csv_path)  # Assumes load_csv is defined
                    if input_df is not None:
                        st.dataframe(input_df.head(), height=200)
                    else:
                        st.error(f"Input CSV not found: {csv_path}")

                st.divider()

                # --- 2. Display Processed Outputs ---
                st.markdown("#### 2. Processed Feature Files")
                col3, col4 = st.columns(Constants.TWO)

                with col3:
                    st.markdown("**Output Audio Features (Preview)**")
                    audio_csv_path = batch_data.get('output_audio_feature_csv_paths', [None])[0]
                    audio_df = load_csv(audio_csv_path)  # Assumes load_csv is defined
                    
                    if audio_df is not None:
                        st.dataframe(audio_df.head(), height=200)
                        st.download_button(
                            "⬇️ Download Audio Features CSV",
                            audio_df.to_csv(index=False),
                            file_name=f"{batch_name}_audio_features.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"Audio feature CSV not found: {audio_csv_path}")

                with col4:
                    st.markdown("**Output Vision Features (Preview)**")
                    vision_csv_path = batch_data.get('output_vision_filtered_csv_path')
                    vision_df = load_csv(vision_csv_path)  # Assumes load_csv is defined
                    
                    if vision_df is not None:
                        st.dataframe(vision_df.head(), height=200)
                        st.download_button(
                            "⬇️ Download Vision Features CSV",
                            vision_df.to_csv(index=False),
                            file_name=f"{batch_name}_vision_filtered.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"Filtered vision CSV not found: {vision_csv_path}")
                
                st.divider()

                # --- 3. Display RF Guidance ---
                st.markdown("#### 3. RF Feature Importance (Audio)")
                rf_str = batch_data.get('rf_guidance')
                
                if rf_str:
                    rf_df = parse_rf_guidance(rf_str)  # Assumes parse_rf_guidance is defined
                    if rf_df is not None:
                        st.dataframe(rf_df, use_container_width=True)
                else:
                    st.info("No RF guidance was generated for this batch.")
                    
                st.divider()
                
                # --- 4. Display Gemini Response ---
                st.markdown("#### 4. Gemini Fusion Analysis (JSON)")
                json_path = batch_data.get('output_gemini_response_json_path')
                gemini_json = load_json(json_path)  # Assumes load_json is defined
                
                if gemini_json:
                    # st.json(gemini_json)

                    json_string = json.dumps(gemini_json, indent=4)
                    
                    # Display the string in a code block with ONE copy button
                    st.code(json_string, language="json")

                    st.download_button(
                        label="⬇️ Download Gemini JSON",
                        data=json_string,
                        file_name=f"{batch_name}_gemini_response.json",
                        mime="application/json"
                    )

                else:
                    st.error(f"Gemini response JSON/TXT not found: {json_path}")

        
        with tabs[-Constants.ONE]:  # Final Summary
            st.subheader("📋 Final Multimodal Summary")

            if st.session_state.final_summary:
                # Clean the summary text from markdown code fences if present
                cleaned_summary = st.session_state.final_summary.strip()
                cleaned_summary = re.sub(r"^```[a-zA-Z]*|```$", "", cleaned_summary, flags=re.MULTILINE).strip()
                cleaned_summary = re.sub(r"^---$", "", cleaned_summary, flags=re.MULTILINE).strip()

                # Display properly rendered Markdown inside an expander
                with st.expander("🩺 View Final Medical Analysis Report", expanded=True):
                    st.markdown(cleaned_summary, unsafe_allow_html=True)

                # Divider
                st.divider()

                # Download options
                st.download_button(
                    "⬇️ Download Summary (Markdown)",
                    cleaned_summary,
                    file_name="final_medical_analysis_report.md",
                    mime="text/markdown"
                )

            else:
                st.warning("⚠️ No final summary generated yet.")

            # JSON results download
            if "multimodal_results" in st.session_state and st.session_state.multimodal_results:
                results_json = json.dumps(
                    st.session_state.multimodal_results,
                    indent=Constants.TWO,
                    default=str
                )

                st.download_button(
                    "⬇️ Download All Results (JSON)",
                    results_json,
                    file_name="multimodal_results.json",
                    mime="application/json"
                )
 
    def user_input_prompt(self):
        """
            This method implements the different type of inputs from user and their relevant parameters
        """
        with st.expander(Constants.USER_INPUTS_PROMPT):
            prompt_col, batch_col = st.columns(Constants.TWO)
            
            with prompt_col:
                self.user_prompt = st.text_area(
                    Constants.USER_PROMPT_STR,
                    """Analyze this data for medical assessment""",
                )
 
            with batch_col:
                if st.session_state.selected_mode == Constants.VISION_STR:
                    vision_batch_size = st.number_input(
                        Constants.VISION_BATCH_STR,
                        value=Constants.ONE,
                        max_value=self.video_duration if self.video_duration else 10,
                        key=Constants.VISION_BATCH_KEY,
                        help="Number of seconds per batch"
                    )
                    self.vision_batch_size = int(vision_batch_size * self.video_fps)
                    if self.uploaded_file and self.frame_count:
                        self.total_batches = math.ceil(self.frame_count / self.vision_batch_size)
                    
                elif st.session_state.selected_mode == Constants.AUDIO_STR:
                    audio_batch_size = st.number_input(
                        Constants.AUDIO_BATCH_STR,
                        value=Constants.ONE,
                        max_value=self.audio_duration if self.audio_duration else 10,
                        key=Constants.AUDIO_BATCH_KEY,
                        help="Audio batch duration in seconds"
                    )
                    self.audio_batch_size = audio_batch_size
                    
                    self.audio_batch_size = audio_batch_size
                    
                    audio_overlap_size = st.number_input(
                        Constants.AUDIO_OVERLAP_STR,
                        value=float(Constants.ZERO),
                        max_value=max(float(Constants.ZERO), audio_batch_size - Constants.POINT_ONE) if audio_batch_size > Constants.POINT_ONE else float(Constants.ZERO),
                        step=Constants.HALF,
                        key=Constants.AUDIO_OVERLAP_KEY,
                        help="Overlap between audio batches in seconds"
                    )
                    self.audio_overlap = audio_overlap_size
                    
                    audio_top_features = st.number_input(
                        Constants.AUDIO_TOP_STR,
                        value=Constants.FIVE,
                        min_value=Constants.ONE,
                        max_value=24,
                        key=Constants.AUDIO_TOP_KEY,
                        help="Number of top acoustic features to extract"
                    )
                    if self.uploaded_file is not None:
                        self.audio_top_features = audio_top_features
                        step = max(Constants.EPSILON, float(self.audio_batch_size) - float(self.audio_overlap))
                        expected = Constants.ONE + math.floor(max(float(Constants.ZERO), float(self.audio_duration) - float(self.audio_batch_size)) / step)
                        st.session_state.expected_total_batches = int(max(Constants.ONE, expected))

                    
                elif st.session_state.selected_mode == Constants.MULTIMODAL_STR:
                    self.multimodal_batch_size = st.number_input(
                        Constants.MULTI_BATCH_STR,
                        value=Constants.ONE,
                        max_value=self.video_duration if self.video_duration else 10,
                        key=Constants.MULTI_BATCH_KEY
                    )
                    self.multi_overlap_size = st.number_input(
                        Constants.MULTI_OVERLAP_STR,
                        value=Constants.ONE,
                        key=Constants.MULTI_OVERLAP_KEY
                    )
                    self.multi_top_features = st.number_input(
                        Constants.MULTI_TOP_STR,
                        value=Constants.FIVE,
                        key=Constants.MULTI_TOP_KEY
                    )
 
 
    def agent_calling(self):
        """
            This method calls the appropriate agent based on selected mode
        """
        if st.session_state.selected_mode == Constants.VISION_STR:
            st.session_state.processing_complete = False
            file_ext = self.uploaded_file.name.split(Constants.DOT)[-Constants.ONE].lower()
            os.makedirs(Constants.VISION_OUT_DIR, exist_ok=True)
            if file_ext in Constants.VIDEO_EXT:
                # Video processing
                video_path = os.path.join(Constants.VISION_OUT_DIR, "input_video.mp4")
                with open(video_path, Constants.WRITE_BINARY) as f:
                    f.write(self.uploaded_file.getvalue())
                
                st.info("🎬 Video processing: Extracting blendshapes...")
                csv_path = Infer(video_path, output_dir=Constants.VISION_OUT_DIR).inference()
                self.df = pd.read_csv(csv_path)    
            elif file_ext in Constants.CSV_EXT:  
                pass
            else:
                st.error(f"Unsupported file type: {file_ext}")
                return
            
            # FIXED: Pass correct parameters with raw file bytes
            self.vision_medical_agent._run_processing_pipeline(
                self.df,                            # extracted data
                work_dir=Constants.VISION_OUT_DIR,  # Output directory
                batch_size=self.vision_batch_size,  # Frames per batch
                user_prompt=self.user_prompt        # Analysis prompt
            )
            
        elif st.session_state.selected_mode == Constants.AUDIO_STR:
            st.session_state.audio_processing = True
            st.session_state.processing_complete = False
            st.info("🎧 Starting audio analysis...")
        
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(self.uploaded_file.read())
                audio_path = tmp_audio.name
        
            progress_bar = st.progress(Constants.ZERO)
            status_text = st.empty()
        
            
            def progress_callback(batches):
                st.session_state.audio_batch_data = batches
                done = len(batches)
                total = max(Constants.ONE, st.session_state.get('expected_total_batches', done))
                percent = min(Constants.HUNDERD, int(done / total * Constants.HUNDERD))
                progress_bar.progress(percent)

        
            def status_callback(msg, progress):
                status_text.info(f"{msg} ({progress}%)")
        
            async def run_audio_agent():
                return await self.audio_agent.process_audio_async(
                    audio_path=audio_path,
                    audio_file_name = self.audio_file_name,
                    batch_seconds=self.audio_batch_size,
                    overlap_seconds=self.audio_overlap,
                    num_features=self.audio_top_features,
                    user_prompt=self.user_prompt,
                    progress_callback=progress_callback,
                    status_callback=status_callback
                )
            
            try:
                result = run_coro(run_audio_agent())

                # ✅ Write results
                st.session_state.final_summary = result.get("summary_text", "")
                st.session_state.audio_batch_data = result.get("batches", [])
                st.session_state.processing_complete = True

                # ✅ Mark completion (used by UI to know these are fresh)
                st.session_state.last_completed_at = time.time()

                st.success("✅ Audio processing complete!")

                # ✅ Immediately refresh the page so results are visible *now*
                st.rerun()

            except Exception as e:
                st.error(f"Audio processing failed: {e}")
            finally:
                st.session_state.audio_processing = False
                progress_bar.empty()
                status_text.empty()

        elif st.session_state.selected_mode == Constants.MULTIMODAL_STR:
            st.session_state.multimodal_processing = True
            st.session_state.processing_complete = False
            st.info("🎬🎧 Starting multimodal analysis...")

            multimodal_output_dir = Constants.DECODED_FILES_DIR
            os.makedirs(multimodal_output_dir, exist_ok=True)

            progress_bar = st.progress(Constants.ZERO)
            status_text = st.empty()

            async def run_multimodal_pipeline():
                try:
                    # CASE 1: User uploaded a video file
                    if st.session_state.multi_video_type == Constants.VIDEO_FILES_STR:
                        status_text.info("🎞️ Detected video input — extracting audio and vision features...")
                        progress_bar.progress(Constants.TEN)

                        # Save uploaded video
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                            tmp_video.write(self.uploaded_file.read())
                            video_path = tmp_video.name

                        # Step 1: Extract audio from video
                        status_text.info("🎵 Extracting audio from video...")
                        master_audio_path = await extract_audio_from_video(
                            Path(video_path), multimodal_output_dir
                        )
                        if not master_audio_path or not Path(master_audio_path).exists():
                            raise FileNotFoundError("Audio extraction failed")
                        st.session_state.multimodal_results['audio_path'] = str(master_audio_path)
                        progress_bar.progress(30)

                        # Step 2: Extract facial features (CSV)
                        status_text.info("👁️ Extracting facial features from video...")
                        infer_obj = Infer(video_path, output_dir=multimodal_output_dir)
                        master_csv_path = infer_obj.inference()
                        if not Path(master_csv_path).exists():
                            raise FileNotFoundError(f"Vision CSV not created at: {master_csv_path}")
                        st.session_state.multimodal_results['csv_path'] = master_csv_path
                        progress_bar.progress(50)

                    # CASE 2: User uploaded CSV + Audio files separately
                    elif st.session_state.multi_video_type == Constants.VISION_CSV_AND_AUDIO:
                        status_text.info("📂 Detected CSV + Audio inputs — using directly for analysis...")
                        progress_bar.progress(Constants.TWENTY)

                        # Save CSV file
                        original_vision_suffix = Path(self.uploaded_file.name).suffix
                        # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=original_vision_suffix) as tmp_csv:
                            tmp_csv.write(self.uploaded_file.read())
                            master_csv_path = tmp_csv.name
                        st.session_state.multimodal_results['csv_path'] = master_csv_path



                        original_audio_suffix = Path(self.audio_file.name).suffix

                        # print('##########################')
                        # print(original_audio_suffix)
                        # Save Audio file
                        # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=original_audio_suffix) as tmp_audio:
                            tmp_audio.write(self.audio_file.read())
                            master_audio_path = tmp_audio.name
                        st.session_state.multimodal_results['audio_path'] = master_audio_path
                        progress_bar.progress(40)

                    else:
                        raise ValueError("Unsupported multimodal input type selected.")

                    # Step 3: Run multimodal fusion
                    status_text.info("🔄 Running multimodal fusion pipeline...")
                    progress_bar.progress(70)

                    pipeline_result = await run_full_pipeline(
                        user_prompt=self.user_prompt,
                        master_csv_path=str(master_csv_path),
                        master_audio_path=str(master_audio_path),
                        batch_duration_seconds=self.multimodal_batch_size,
                        overlap_duration=self.multi_overlap_size
                    )

                    progress_bar.progress(95)

                    # Store results
                    st.session_state.multimodal_results.update({
                        'pipeline_result': pipeline_result,
                        'output_dir': str(multimodal_output_dir)
                    })

                    if pipeline_result:

                        summary_path_str = pipeline_result.get('final_summary_report_path')
                        if summary_path_str:
                            try:
                                summary_path = Path(summary_path_str)
                                if summary_path.exists():
                                    # Read the text from the .txt file
                                    st.session_state.final_summary = summary_path.read_text(encoding='utf-8')

                                else:
                                    st.session_state.final_summary = f"Error: Summary file not found at {summary_path_str}"
                            except Exception as e:
                                st.session_state.final_summary = f"Error reading summary file: {e}"

                        else:
                            st.session_state.final_summary = "No final summary report was generated."

                        
                            
                        st.session_state.multimodal_batch_data = pipeline_result.get('batches', {})
                    

                    status_text.success("✅ Multimodal processing complete!")
                    progress_bar.progress(100)
                    st.session_state.processing_complete = True
                    st.session_state.last_completed_at = time.time()

                    return pipeline_result

                except Exception as e:
                    status_text.error(f"❌ Multimodal processing failed: {str(e)}")
                    st.error(f"Error details: {str(e)}")
                    raise

            # Run pipeline
            try:
                result = run_coro(run_multimodal_pipeline())
                st.success("✅ Multimodal analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Multimodal processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                st.session_state.multimodal_processing = False
                progress_bar.empty()
                status_text.empty()

 
    def processing_flow(self):
        """
            This method implements the processing flow with batch tracking
        """
        # Only show if we have batches or are processing
        if not st.session_state.batch_data and not st.session_state.get('currently_processing', False):
            return
        
        st.divider()
        
        # Calculate metrics
        total_batches = len(st.session_state.batch_data)
        completed = total_batches
        currently_processing = st.session_state.get('current_batch_index', 0)
        
        # If still processing, show expected total
        expected_total = st.session_state.get('expected_total_batches', total_batches)
        processing = st.session_state.get('currently_processing', False)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(Constants.FIVE)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Expected Batches", expected_total)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Completed", completed, delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if processing:
                st.metric("Currently Processing", f"Batch {currently_processing + Constants.ONE}")
            else:
                st.metric("Status", "Done" if st.session_state.processing_complete else "Idle")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            remaining = expected_total - completed
            st.metric("Remaining", remaining)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            completion_rate = (completed / expected_total * Constants.HUNDERD) if expected_total > 0 else 0
            st.metric("Progress", f"{completion_rate:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
 
    def results_pane(self):
        """
            This method implements the results of the processing
        """
        if not st.session_state.processing_complete and not self.process_button:
            st.info("👆 Upload a file, configure parameters, and click 'Start Analysis' to begin.")
            return
        
        result_tab, = st.tabs([st.session_state.selected_mode])
        
        if st.session_state.selected_mode == Constants.VISION_STR:
            with result_tab:
                self.vision_ui()
                
        elif st.session_state.selected_mode == Constants.AUDIO_STR:
            with result_tab:
                self.audio_ui()
                
        elif st.session_state.selected_mode == Constants.MULTIMODAL_STR:
            with result_tab:
                self.multimodel_ui()
        
        st.markdown('</div>', unsafe_allow_html=True)
 
        
 
    def ui_content(self):
        """
            This method calls all the UI methods
        """
        # APP Title  
        st.markdown(f'<div class="main-header">🎬 {Constants.APP_NAME}</div>', unsafe_allow_html=True)
 
        # MODE SELECTIONS AND FILE UPLOAD
        self.input_file_upload()
 
        # PROCESSING FLOW METRICS
        self.processing_flow()
 
        # RESULTS
        self.results_pane()
 
 
    def ui_content_updates(self):
        """
            This method implements the UI updates as processing happens
        """
        if self.process_button and self.uploaded_file:
            # Clear previous results
            st.session_state.batch_data = []
            st.session_state.final_summary = Constants.INVERTED_STRING
            st.session_state.processing_complete = False
            
            st.session_state.audio_batch_data = []
            st.session_state.last_completed_at = None

            # Start processing
            self.agent_calling()
            
            # After processing completes, rerun to show results
            if st.session_state.processing_complete:
                st.rerun()