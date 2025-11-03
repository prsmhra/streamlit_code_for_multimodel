
import re
import cv2
import tempfile
import os
from pydub import AudioSegment
import math
import streamlit as st
import json
import librosa
import time
from pprint import pprint
import hashlib

 
from src.python.app.constants.constants import Constants
from src.python.app.utils.show_input_file import *
from src.python.app.utils.agents_logs import *
from src.python.app.utils.data_utils import *
from src.python.app.utils.thread_pool_executer import *
from src.python.app.utils.extract_josn_from_text import *
 
from src.python.app.common.vision_agent_call import MedicalAIAgentApp
from src.python.app.common.audio_agent_integration import AudioAgentPipeline
from src.python.app.video_frame_extractor.csv_sav_inference import Infer 

 
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
        
        # Audio-specific session state
        if 'audio_chunk_results' not in st.session_state:
            st.session_state.audio_chunk_results = []
        if 'audio_processing' not in st.session_state:
            st.session_state.audio_processing = False
        if 'audio_batch_data' not in st.session_state:
            st.session_state.audio_batch_data = []
 
        ## class variables
        self.user_prompt = None
        self.uploaded_file = None
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
 
        # Single instance of MedicalAIAgentApp
        self.vision_medical_agent = MedicalAIAgentApp()
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
                        key=f"{Constants.VIDEO_UPLOAD_KEY}_{st.session_state.uploaded_files_key}"
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
                self.uploaded_file = st.file_uploader(
                    Constants.VIDEO_UPLOAD_STR,
                    type=Constants.VIDEO_EXT,
                    accept_multiple_files=False,
                    key=f"{Constants.MULTIMODEL_VIDEO_KEY}_{st.session_state.uploaded_files_key}"
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
        if self.uploaded_file:
            with st.expander(Constants.INPUT_SHOW_EXPAND):
                self.input_file_show_container = st.empty()
                audio_container = st.empty()
                
                file_name = self.uploaded_file.name
                file_ext = file_name.split(Constants.DOT)[-Constants.ONE].lower()
                
                if st.session_state.selected_mode == Constants.VISION_STR and file_ext in Constants.CSV_EXT:
                    self.df = show_csv_files(file_name, self.uploaded_file, self.input_file_show_container)
                    self.video_fps = Constants.DEFAULT_FPS
                    self.video_duration = int(self.df[Constants.FRAME_KEY].nunique() / self.video_fps)
                    self.frame_count = self.df[Constants.FRAME_KEY].nunique()
                    # Reset file pointer
                    self.uploaded_file.seek(Constants.ZERO)
                    
                elif st.session_state.selected_mode == Constants.AUDIO_STR:
                    show_audio_waveform(self.uploaded_file, self.input_file_show_container, audio_container)
                    
                elif st.session_state.selected_mode in [Constants.VISION_STR, Constants.MULTIMODAL_STR] and file_ext in Constants.VIDEO_EXT:
                    self.input_file_show_container.video(self.uploaded_file)
                    # Reset file pointer
                    self.uploaded_file.seek(Constants.ZERO)
 
    def vision_ui(self):
        """
            This method implements vision results
        """
        # Check if we have batch data
        num_batches = len(st.session_state.batch_data)
        pprint(st.session_state.batch_data)
        if num_batches == Constants.ZERO and st.session_state.processing_complete:
            st.warning("No batch data available yet. Processing may have failed or not started.")
            return
        
        batch_tabs = st.tabs([Constants.AGENT_LOGS_HEADING] + 
                             [f"{Constants.BATCH_KEY} {i+1}" for i in range(num_batches)] + 
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
                        st.markdown(f"### 📦 Batch {batch_idx + 1}")
                        
                        for log in filtered_logs:
                            agent_name = log[Constants.AGENT_KEY]
                            message = log.get('message', '')
                            timestamp = log.get('timestamp', '')
                            
                            # Determine if success or error
                            is_success = '✅' in message or 'complete' in message.lower() or 'success' in message.lower()
                            is_error = '❌' in message or 'error' in message.lower() or 'failed' in message.lower()
                            
                            # Create enhanced log card
                            log_html = f'''
                            <div class="agent-log-card {agent_name}">
                                <div class="agent-log-header">
                                    <div>
                                        <span class="batch-badge" style="background: rgba(106, 13, 173, 0.1); color: #6a0dad;">
                                            Batch {batch_idx + 1}
                                        </span>
                                        <span class="agent-name">{agent_name}</span>
                                    </div>
                                    <div class="agent-timestamp">{timestamp}</div>
                                </div>
                                <div class="agent-message">
                                    {message if len(message) < 300 else message[:297] + '...'}
                                </div>
                            </div>
                            '''
                            
                            st.markdown(log_html, unsafe_allow_html=True)
                        
                        st.divider()
                
                with st.expander(f"📋 {Constants.AGENT_EXPENDER_KEY}"):
                    legend_cols = st.columns(Constants.THREE)
                    agent_colors = {
                        'MetaIntentTool': {'bg': '#FFE5E5', 'text': '#8B0000', 'border': '#FF6B6B'},
                        'LlmOrchestrator': {'bg': '#E5F3FF', 'text': '#0B3BA1', 'border': '#4DABF7'},
                        'FrameSamplerTool': {'bg': '#E5FCFF', 'text': '#004E89', 'border': '#74C0FC'},
                        'FramePrefilterTool': {'bg': '#F0E5FF', 'text': '#5F00B2', 'border': '#B197FC'},
                        'FeaturesSelectionTool': {'bg': '#FFF4E5', 'text': '#995A00', 'border': '#FFD43B'},
                        'SymptomAnalyzerTool': {'bg': '#E5F5F0', 'text': '#0B5F0B', 'border': '#51CF66'}
                    }
                    
                    for idx, (agent, colors) in enumerate(agent_colors.items()):
                        with legend_cols[idx % Constants.THREE]:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {colors['bg']} 0%, {colors['bg']}dd 100%); 
                                        padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0;
                                        border-left: 4px solid {colors['border']}; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <strong style="color: {colors['text']};">{agent}</strong>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info(f"🔭 {Constants.NO_BATCH_KEY}")
 
        # Batch detail tabs
        if len(st.session_state.batch_data) >= Constants.ONE:
            for i, batch_tab in enumerate(batch_tabs[Constants.ONE:-2]):
                with batch_tab:
                    batch_info = st.session_state.batch_data[i]
                    batch_state_data = batch_info['session_state_data']
                    batch_df = batch_info['original_df']
 
                    st.subheader(f"Detailed Analysis for Batch {i+1}")
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
                        st.error(f"Could not load data for batch {i+1}. The CSV structure may be incompatible.")
                        st.info("Expected columns: blendshapes (eyeBlink*, mouth*, etc.) or AUs (AU01, AU02, etc.)")
                        with st.expander("Show available columns"):
                            st.write(batch_df.columns.tolist())
                        continue
 
                    if loaded_data['structure']:
                        with nested_tabs[0]:  # Blendshapes
                            if loaded_data['blendshapes_df'] is not None and not loaded_data['blendshapes_df'].empty:
                                st.markdown("### 🎭 Blendshapes by Facial Region")
                                region_tabs = st.tabs(["👁️ Eyes", "👄 Mouth", "👃 Nose"])
                                for r_idx, (region, r_tab) in enumerate(zip(["eyes", "mouth", "nose"], region_tabs)):
                                    with r_tab:
                                        render_regional_blendshapes(loaded_data['blendshapes_df'], loaded_data['structure'], region, f"b{i}_blend_{r_idx}")
                            else:
                                st.warning("No blendshape data for this batch.")
 
                        with nested_tabs[1]:  # AUs
                            if loaded_data['aus_df'] is not None and not loaded_data['aus_df'].empty:
                                st.markdown("### 🔢 AUs by Facial Region")
                                au_region_tabs = st.tabs(["👁️ Eyes AUs", "👄 Mouth AUs", "👃 Nose AUs"])
                                for r_idx, (region, r_tab) in enumerate(zip(["eyes", "mouth", "nose"], au_region_tabs)):
                                    with r_tab:
                                        render_regional_aus(loaded_data['aus_df'], loaded_data['structure'], region, f"b{i}_au_{r_idx}")
                            else:
                                st.warning("No AU data for this batch.")
                        
                        with nested_tabs[2]:  # Emotions/Pain
                            df_to_render = loaded_data['emotions_df'] if loaded_data['emotions_df'] is not None else batch_df
                            render_emotions_pain(df_to_render, loaded_data['structure'], f"b{i}_emo")
 
                    with nested_tabs[3]:  # Sampling
                        st.subheader("Sampling Results")
                        sampler_raw = batch_state_data.get("csv_sampler_result", "")
                        st.markdown("#### ⏱️ Frame Sampling Analysis")
                        if sampler_raw and isinstance(sampler_raw, dict):
                            st.metric("Target FPS", sampler_raw.get("target_fps", "N/A"))
                            st.info(f"**LLM Reason:** {sampler_raw.get('reason', 'N/A')}")
                        else:
                            st.warning("Sampling was not performed for this batch.")
 
                    with nested_tabs[4]:  # Prefilter
                        st.subheader("Prefilter Agent Results")
                        prefilter_result = batch_state_data.get("prefilter_result")
                        if prefilter_result and isinstance(prefilter_result, dict):
                            st.metric("Useful Frames Found?", "Yes" if prefilter_result.get("useful") else "No")
                            st.info(f"**Identified Frame Ranges:** `{prefilter_result.get('kept_ranges', '[]')}`")
                            st.info(f"**LLM Reason:** {prefilter_result.get('reason', 'N/A')}")
                            st.success(f"Filtered CSV saved to: `{prefilter_result.get('filtered_csv_path')}`")
                        else:
                            st.warning("Prefilter was not run or found no useful frames for this batch.")
 
                    with nested_tabs[5]:  # Filtered Data
                        st.subheader("Region Filter Agent Results")
                        filter_result = batch_state_data.get("filtered_csv_result")
                        if filter_result and isinstance(filter_result, dict):
                            st.info(f"**Identified Active Regions:** `{filter_result.get('regions', '[]')}`")
                            st.success(f"Region-filtered CSV saved to: `{filter_result.get('filtered_csv_path')}`")
                        else:
                            st.warning("Region filter was not run for this batch.")
 
                    with nested_tabs[6]:  # Symptom Analysis
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
                            if symptom_analysis_raw:
                                st.code(symptom_analysis_raw)
                    
                    with nested_tabs[7]:  # Session State
                        st.subheader(f"Full Session State for Batch {i+1}")
                        st.json(batch_state_data)
        
        # MetaIntent Tab
        with batch_tabs[-2]:
            st.subheader("Meta-Intent Analysis")
            if st.session_state.batch_data:
                meta_raw = st.session_state.batch_data[0]['session_state_data'].get("meta_intent_result", "")
                st.markdown("#### 🧩 MetaIntent Classification")
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
        
        # Final Summary Tab
        with batch_tabs[-1]:
            st.header("Overall Clinical Summary Across All Batches")
            if st.session_state.final_summary:
                st.markdown(st.session_state.final_summary)
            else:
                st.warning("The final summary could not be generated or is empty.")
 

    
    def render_audio_json_result(self, result_text, idx):
        """
        Render Gemini analysis:
        - If JSON (dict or JSON string, possibly fenced/mixed), show structured UI.
        - Else, render the original dark card with raw text.
        """
        # 1) Parse JSON robustly (handles ```json fences and extra prose)
        data = None
        if isinstance(result_text, dict):
            data = result_text
        else:
            candidate = strip_code_fence(result_text)
            try:
                data = json.loads(candidate)
            except Exception:
                sliced = extract_braced_json(candidate)
                if sliced:
                    try:
                        data = json.loads(sliced)
                    except Exception:
                        data = None

        st.markdown("### Gemini Analysis Result")

        # 2) Fallback: original dark card if not JSON
        if not data:
            st.markdown(f"""
            <div class='response-card' style='background:#071127; color:#ecfeff; padding:16px; border-radius:12px;'>
            <pre style='margin:0; white-space:pre-wrap; word-wrap:break-word;'>{result_text}</pre>
            </div>
            """, unsafe_allow_html=True)
            return
        # 3) Styles
        st.markdown("""
        <style>
        .card { background:#0b1739; color:#ecfeff; border-radius:12px; padding:16px; border:1px solid #163058; }
        .subcard { background:#0e1c48; color:#ebf4ff; border-radius:10px; padding:12px; border:1px solid #1c3666; margin-top:10px; }
        .badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }
        .badge-ok { background:#133a2c; color:#9ff0c1; border:1px solid #2b6f50; }
        .badge-warn { background:#3a2b13; color:#f0d49f; border:1px solid #6f502b; }
        .muted { color:#cbe0ff; opacity:0.9; }
        .divider { height:1px; background:#193256; margin:12px 0; border-radius:1px; }
        .code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
        </style>
        """, unsafe_allow_html=True)

        # 4) Summary & Confidence
        summary = to_text(data.get("summary", "No summary provided."))
        confidence = to_text(data.get("confidence_assessment", ""))
        conf_class = "badge-ok" if "high" in confidence.lower() else "badge-warn" if confidence else "badge-warn"

        st.markdown(f"""
        <div class="card">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
            <div style="font-weight:700; font-size:18px;">Overall Summary</div>
            <span class="badge {conf_class}">{confidence or "Confidence: N/A"}</span>
        </div>
        <div class="divider"></div>
        <div class="muted">{summary}</div>
        </div>
        """, unsafe_allow_html=True)

        # 5) Per‑speaker (robust to strings/non-dicts)
        per_speaker_list = normalize_per_speaker(data.get("per_speaker_findings", {}))
        if per_speaker_list:
            st.subheader("Per-speaker findings")
            for entry in per_speaker_list:
                speaker = entry.get("speaker", "Unknown")
                conclusions = entry.get("conclusions", "N/A")
                evidence = entry.get("evidence", "N/A")
                st.markdown(f"""
                <div class="subcard">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span class="badge badge-ok">{speaker}</span>
                    <span class="badge badge-warn">Findings</span>
                </div>
                <div style="font-weight:600; margin-bottom:6px;">Conclusions</div>
                <div>{conclusions}</div>
                <div class="divider"></div>
                <div style="font-weight:600; margin-bottom:6px;">Evidence</div>
                <div class="code">{evidence}</div>
                </div>
                """, unsafe_allow_html=True)

        # 6) Evidence list (robust)
        ev_list = normalize_evidence_list(data.get("evidence_list", []))
        if ev_list:
            st.subheader("Detailed Evidence")
            for idx, item in enumerate(ev_list, 1):
                title = f"Evidence {idx}: {item['speaker']} • {item['window_length_s']}s • Frame {item['frame_index']}"
                with st.expander(title):
                    features = item.get("features", {})
                    if isinstance(features, dict):
                        features_text = json.dumps(features, indent=2, ensure_ascii=False)
                    else:
                        features_text = to_text(features)
                    st.markdown(f"""
                    <div class="subcard">
                    <div style="font-weight:600; margin-bottom:6px;">Features</div>
                    <div class="code">{features_text}</div>
                    <div class="divider"></div>
                    <div style="font-weight:600; margin-bottom:6px;">Top Random‑Forest Features</div>
                    <div class="code">{item.get("top_rf_features", "N/A")}</div>
                    <div class="divider"></div>
                    <div style="font-weight:600; margin-bottom:6px;">Acoustic Claim</div>
                    <div>{item.get("acoustic_claim", "N/A")}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # 7) Download + Raw JSON
        pretty = json.dumps(data, indent=Constants.TWO, ensure_ascii=False)

        digest = hashlib.md5(pretty.encode("utf-8")).hexdigest()[:Constants.TEN]
        dl_key = f"{Constants.AUDIO_STR.lower()}{Constants.UNDERSCORE}{idx}{Constants.UNDERSCORE}{digest}"

        st.download_button("⬇️ Download JSON", pretty,
                        file_name="gemini_audio_analysis.json",
                        mime="application/json",
                        key = dl_key,
                        use_container_width=True)
        with st.expander("Raw JSON"):
            st.json(data)

    def audio_ui(self):
        """
            This method implements audio processing results
        """
        num_batches = len(st.session_state.audio_batch_data)
        if num_batches == Constants.ZERO and st.session_state.processing_complete:
            st.warning("No audio batch data available yet. Processing may have failed or not started.")
            return
        
        batch_tabs = st.tabs([Constants.AGENT_LOGS_HEADING] + 
                             [f"Audio Batch {i+1}" for i in range(num_batches)] + 
                             [Constants.FINAL_SUMMERY_KEY])
        
        with batch_tabs[Constants.ZERO]:
            st.subheader("Audio Processing Logs")
            if len(st.session_state.audio_batch_data) > Constants.ZERO:
                for i, batch in enumerate(st.session_state.audio_batch_data):
                    with st.expander(f"Audio Batch {i+1} - {batch.get('start_s', 0):.2f}s to {batch.get('start_s', 0) + batch.get('duration_s', 0):.2f}s"):
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
                st.info(f"Time range: {batch_info.get('start_s', Constants.ZERO):.2f}s - {batch_info.get('start_s', 0) + batch_info.get('duration_s', 0):.2f}s")
                
                if batch_info.get('error'):
                    st.error(f"Processing failed: {batch_info['error']}")
                else:
                    result_text = batch_info.get('result', 'No result available')
                    
                    
                    self.render_audio_json_result(result_text, i)
                    
                    # Show batch directory
                    if batch_info.get('batch_dir'):
                        st.markdown(f"**Output Directory:** `{batch_info['batch_dir']}`")
        
        # Final Summary Tab
        with batch_tabs[-1]:
            st.header("Overall Audio Analysis Summary")
            if st.session_state.final_summary:
                st.markdown(st.session_state.final_summary)
            else:
                st.warning("The final audio summary could not be generated or is empty.")
        
    def multimodel_ui(self):
        st.info("Multimodal processing not yet implemented")
 
 
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
                        step = max(1e-6, float(self.audio_batch_size) - float(self.audio_overlap))
                        expected = 1 + math.floor(max(0.0, float(self.audio_duration) - float(self.audio_batch_size)) / step)
                        st.session_state.expected_total_batches = int(max(1, expected))

                    
                elif st.session_state.selected_mode == Constants.MULTIMODAL_STR:
                    multimodel_batch_size = st.number_input(
                        Constants.MULTI_BATCH_STR,
                        value=Constants.ONE,
                        max_value=self.video_duration if self.video_duration else 10,
                        key=Constants.MULTI_BATCH_KEY
                    )
                    multi_overlap_size = st.number_input(
                        Constants.MULTI_OVERLAP_STR,
                        value=Constants.ONE,
                        key=Constants.MULTI_OVERLAP_KEY
                    )
                    multi_top_features = st.number_input(
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
                csv_path = Infer(video_path).inference()
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
        
            progress_bar = st.progress(0)
            status_text = st.empty()
        
            
            def progress_callback(batches):
                st.session_state.audio_batch_data = batches
                done = len(batches)
                total = max(1, st.session_state.get('expected_total_batches', done))
                percent = min(100, int(done / total * 100))
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

            # try:
            #     result = run_coro(run_audio_agent())
            #     st.session_state.final_summary = result.get("summary_text", "")
            #     st.session_state.audio_batch_data = result.get("batches", [])
            #     st.session_state.processing_complete = True
            #     st.success("✅ Audio processing complete!")
            # except Exception as e:
            #     st.error(f"Audio processing failed: {e}")
            # finally:
            #     st.session_state.audio_processing = False
            #     progress_bar.empty()
            #     status_text.empty()
            
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
            st.warning("Multimodal processing not yet implemented", icon="⚠️")
            
        else:
            st.error("WRONG MODE SELECTED", icon="⚠️")
 
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
                st.metric("Currently Processing", f"Batch {currently_processing + 1}")
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
            completion_rate = (completed / expected_total * 100) if expected_total > 0 else 0
            st.metric("Progress", f"{completion_rate:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Batch Timeline
        if expected_total > Constants.ZERO:
            st.markdown("### 📊 Batch Processing Timeline")
            
            timeline_html = '<div class="batch-timeline">'
            
            for i in range(expected_total):
                if i < completed:
                    status_class = "completed"
                    icon = "✅"
                    status_text = "Completed"
                    color = "#27ae60"
                elif i == currently_processing and processing:
                    status_class = "processing"
                    icon = "⚙️"
                    status_text = "Processing..."
                    color = "#f39c12"
                else:
                    status_class = "pending"
                    icon = "⏳"
                    status_text = "Pending"
                    color = "#95a5a6"
                
                timeline_html += f"""
                    <div class="batch-timeline-item {status_class}">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="font-size: 1.5em;">{icon}</span>
                            <div>
                                <div style="font-weight: bold; color: {color};">Batch {i + 1}</div>
                                <div style="font-size: 0.9em; color: #666;">{status_text}</div>
                            </div>
                        </div>
                    </div>
                """
            timeline_html += '</div>'
            st.markdown(timeline_html, unsafe_allow_html=True)
        
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