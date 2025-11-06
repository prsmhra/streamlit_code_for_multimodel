import os
from pathlib import Path

class Constants:
    """
    Centralized constants for the Multimodal Medical Agent System
    Organized by functional categories for better maintainability
    """
    
    # ============================================================================
    # APPLICATION METADATA
    # ============================================================================
    APP_NAME = "Multimodal Medical Agent System"
    USER_ID = "streamlit_user"
    INFO = "INFO"
    
    # ============================================================================
    # PROCESSING MODES & UI STRINGS
    # ============================================================================
    VISION_STR = "Vision"
    AUDIO_STR = "Audio"
    MULTIMODAL_STR = "Multimodal"
    SELECTION_LIST = [MULTIMODAL_STR, VISION_STR, AUDIO_STR]
    
    MODEL_SELECTION = "### 🎯 Select Processing Mode"
    MODEL_SELECTION_STR = "Select Processing Mode"
    SELECTION_HELP_STR = "Choose the type of processing you want to perform"
    SELECTION_KEY = "processing_mode"
    INPUT_TYPE_STR = "Input Type"
    USER_PROMPT_STR = "User Prompt for Analysis"
    USER_INPUTS_PROMPT = "User Input and Prompt"
    INPUT_SHOW_EXPAND = "Visualize the Input"
    VISION_INPUT_TYPE = "vision_input_type"
    
    # ============================================================================
    # VISION PROCESSING
    # ============================================================================
    # Vision Input Types
    VIDEO_FILES_STR = "Video Files"
    CSV_FILES_STR = "CSV Data"
    VISION_INPUT_TYPE_LIST = [VIDEO_FILES_STR, CSV_FILES_STR]
    VISION_TYPE_STR = "vision_type"
    VISION_HEADING = "### 🎬 Vision Processing"
    
    # MediaPipe Configuration
    MP_TASK_FILE = "assets/face_landmarker.task"
    NOSE_POINT = 8
    CHIN_POINT = 152
    EYE_POINT_L = 33
    EYE_POINT_R = 263
    
    # Video Upload
    VIDEO_UPLOAD_STR = "Upload Video Files"
    VIDEO_EXT = ['mp4', 'avi', 'mov', 'mkv']
    MP4 = ".mp4"
    VIDEO_UPLOAD_KEY = "video_upload"
    
    # CSV Upload
    CSV_UPLOAD_STR = "Upload CSV Files"
    CSV_EXT = ['csv', 'xlsx']
    CSV_UPLOAD_KEY = "csv_upload"
    CSV_EXTENSION = ".csv"
    
    # ============================================================================
    # AUDIO PROCESSING
    # ============================================================================
    AUDIO_HEADING = "### 🎵 Audio Processing"
    AUDIO_UPLOAD_STR = "Upload Audio Files"
    AUDIO_EXT = ['mp3', 'wav', 'flac', 'm4a', 'aac']
    AUDIO_UPLOAD_KEY = "audio_upload"
    AUDIO_FORMAT = "audio/wav"
    WAV = "wav"
    MP3 = "mp3"
    
    # Audio Frame Configuration
    AUDIO_FRAME_SIZE = (10, 2)
    X_LABEL = "Time (s)"
    Y_LABEL = "Amplitude"
    
    # ============================================================================
    # MULTIMODAL PROCESSING
    # ============================================================================
    MULTIMODEL_HEADING = "### 🔄 Multimodal Processing"
    MULTIMODEL_VIDEO_KEY = "multi_video"
    CSV_KEY = "csv"
    VIDEO_KEY = "video"
    
    # ============================================================================
    # HYPERPARAMETERS & BATCH CONFIGURATION
    # ============================================================================
    # Vision Batch Parameters
    VISION_BATCH_STR = "Batch Size(seconds)"
    VISION_BATCH_KEY = "vision_batch"
    
    # Audio Batch Parameters
    AUDIO_BATCH_STR = "Batch Size(seconds)"
    AUDIO_BATCH_KEY = "audio_batch"
    AUDIO_OVERLAP_STR = "Batch Overlap(seconds)"
    AUDIO_OVERLAP_KEY = "audio_overlap"
    AUDIO_TOP_STR = "Top Audio Features"
    AUDIO_TOP_KEY = "audio_top_features"
    
    # Multimodal Batch Parameters
    MULTI_BATCH_STR = "Batch Size(seconds)"
    MULTI_BATCH_KEY = "multi_batch"
    MULTI_OVERLAP_STR = "Batch Overlap(seconds)"
    MULTI_OVERLAP_KEY = "multi_overlap"
    MULTI_TOP_STR = "Top Audio Features"
    MULTI_TOP_KEY = "multi_top_features"
    
    # ============================================================================
    # SESSION STATE KEYS
    # ============================================================================
    BATCHES_STR = "batches"
    PROCESSING_STR = "processing"
    CURRENT_BATCH_STR = "current_batch"
    SELECTED_MODE = "selected_mode"
    UPLOADED_FILE_KEY = "uploaded_files_key"
    WIDE_STR = "wide"
    UPLOADED_FILES = "uploaded_files"
    
    # ============================================================================
    # AGENT NAMES & TOOLS
    # ============================================================================
    # Agent Names
    METAINTENTLLM_KEY = "MetaIntentLLM"
    METAINTENTTOOL_KEY = "MetaIntentTool"
    PREFILTER_AGENT = "PrefilterLLM"
    SAMPLINGFRAME_AGENT = "SamplingFramesLLM"
    REGIONDETECTOR_AGENT = "RegionDetectorLLM"
    SYMPTOMANALYZER_AGENT = "SymptomAnalyzerLLM"
    ORCHESTRATOR_AGENT = "PipelineOrchestratorLLM"
    SYMPTOM_REASONER_AGENT = "SymptomReasonerLLM"
    LLM_ORCHESTRATOR = "LlmOrchestrator"
    
    # Tool Names
    FRAME_PREFILTER_TOOL = "FramePrefilterTool"
    CSV_FILTER_TOOL = "CSVFilterTool"
    SYMPTOMANALYZER_TOOL = "SymptomAnalyzerTool"
    FEATURE_SELECTION_TOOL = "FeaturesSelectionTool"
    FRAMESAMPLER_TOOL = "FrameSamplerTool"
    METAINTENT_TOOL = "MetaIntentTool"
    
    # Agent Color Configuration
    AGENT_COLORS = {
        "MetaIntentLLM": {"bg": "#FFE5E5", "border": "#FF6B6B", "text": "#8B0000"},
        "MetaIntentTool": {"bg": "#FFE5E5", "border": "#FF6B6B", "text": "#8B0000"},
        "PipelineOrchestratorLLM": {"bg": "#E5F3FF", "border": "#4DABF7", "text": "#0B3BA1"},
        "LlmOrchestrator": {"bg": "#E5F3FF", "border": "#4DABF7", "text": "#0B3BA1"},
        "FrameSamplerTool": {"bg": "#E5FCFF", "border": "#74C0FC", "text": "#004E89"},
        "SamplingFramesLLM": {"bg": "#E5FCFF", "border": "#74C0FC", "text": "#004E89"},
        "FramePrefilterTool": {"bg": "#F0E5FF", "border": "#B197FC", "text": "#5F00B2"},
        "PrefilterLLM": {"bg": "#F0E5FF", "border": "#B197FC", "text": "#5F00B2"},
        "FeaturesSelectionTool": {"bg": "#FFF4E5", "border": "#FFD43B", "text": "#995A00"},
        "RegionDetectorLLM": {"bg": "#FFF4E5", "border": "#FFD43B", "text": "#995A00"},
        "SymptomAnalyzerTool": {"bg": "#E5F5F0", "border": "#51CF66", "text": "#0B5F0B"},
        "SymptomReasonerLLM": {"bg": "#E5F5F0", "border": "#51CF66", "text": "#0B5F0B"},
    }
    
    # ============================================================================
    # AGENT LOGS & UI KEYS
    # ============================================================================
    AGENT_LOGS_HEADING = "Agent Execution Logs"
    AGENT_LOGS_KEY = "agent_logs"
    BATCH_ID_KEY = "batch_id"
    UNKNOWN = 'unknown'
    BATCH_KEY = "batch"
    BATCH_DATA_KEY = 'batch_data'
    AGENTS_LOGS_KEY = 'agent_logs'
    AGENT_KEY = "agent"
    GLOBAL_AGENT_FILTER_KEY = "Global Agent Filter"
    RESET_FILTER_KEY = "Reset Filters"
    MULTISELECT_KEY = "Show agents across all batches:"
    AGENT_EXPENDER_KEY = "Agent Color Legend"
    BG_KEY = "bg"
    TEXT_KEY = "text"
    NO_BATCH_KEY = "No batch data available."
    
    # ============================================================================
    # FACIAL FEATURES - BLENDSHAPES
    # ============================================================================
    # Blendshape Keys
    BROW_DOWN_LEFT_KEY = "browDownLeft"
    BROW_DOWN_RIGHT_KEY = "browDownRight"
    BROW_INNER_UP_KEY = "browInnerUp"
    BROW_OUTER_UP_LEFT_KEY = "browOuterUpLeft"
    BROW_OUTER_UP_RIGHT_KEY = "browOuterUpRight"
    CHEEK_PUFF_KEY = "cheekPuff"
    CHEEK_SQUINT_LEFT_KEY = "cheekSquintLeft"
    CHEEK_SQUINT_RIGHT_KEY = "cheekSquintRight"
    EYE_BLINK_LEFT_KEY = "eyeBlinkLeft"
    EYE_BLINK_RIGHT_KEY = "eyeBlinkRight"
    EYE_LOOK_DOWN_LEFT_KEY = "eyeLookDownLeft"
    EYE_LOOK_DOWN_RIGHT_KEY = "eyeLookDownRight"
    EYE_LOOK_IN_LEFT_KEY = "eyeLookInLeft"
    EYE_LOOK_IN_RIGHT_KEY = "eyeLookInRight"
    EYE_LOOK_OUT_LEFT_KEY = "eyeLookOutLeft"
    EYE_LOOK_OUT_RIGHT_KEY = "eyeLookOutRight"
    EYE_LOOK_UP_LEFT_KEY = "eyeLookUpLeft"
    EYE_LOOK_UP_RIGHT_KEY = "eyeLookUpRight"
    EYE_SQUINT_LEFT_KEY = "eyeSquintLeft"
    EYE_SQUINT_RIGHT_KEY = "eyeSquintRight"
    EYE_WIDE_LEFT_KEY = "eyeWideLeft"
    EYE_WIDE_RIGHT_KEY = "eyeWideRight"
    JAW_FORWARD_KEY = "jawForward"
    JAW_LEFT_KEY = "jawLeft"
    JAW_OPEN_KEY = "jawOpen"
    JAW_RIGHT_KEY = "jawRight"
    MOUTH_CLOSE_KEY = "mouthClose"
    MOUTH_DIMPLE_LEFT_KEY = "mouthDimpleLeft"
    MOUTH_DIMPLE_RIGHT_KEY = "mouthDimpleRight"
    MOUTH_FROWN_LEFT_KEY = "mouthFrownLeft"
    MOUTH_FROWN_RIGHT_KEY = "mouthFrownRight"
    MOUTH_FUNNEL_KEY = "mouthFunnel"
    MOUTH_LEFT_KEY = "mouthLeft"
    MOUTH_LOWER_DOWN_LEFT_KEY = "mouthLowerDownLeft"
    MOUTH_LOWER_DOWN_RIGHT_KEY = "mouthLowerDownRight"
    MOUTH_PRESS_LEFT_KEY = "mouthPressLeft"
    MOUTH_PRESS_RIGHT_KEY = "mouthPressRight"
    MOUTH_PUCKER_KEY = "mouthPucker"
    MOUTH_RIGHT_KEY = "mouthRight"
    MOUTH_ROLL_LOWER_KEY = "mouthRollLower"
    MOUTH_ROLL_UPPER_KEY = "mouthRollUpper"
    MOUTH_SHRUG_LOWER_KEY = "mouthShrugLower"
    MOUTH_SHRUG_UPPER_KEY = "mouthShrugUpper"
    MOUTH_SMILE_LEFT_KEY = "mouthSmileLeft"
    MOUTH_SMILE_RIGHT_KEY = "mouthSmileRight"
    MOUTH_STRETCH_LEFT_KEY = "mouthStretchLeft"
    MOUTH_STRETCH_RIGHT_KEY = "mouthStretchRight"
    MOUTH_UPPER_UP_LEFT_KEY = "mouthUpperUpLeft"
    MOUTH_UPPER_UP_RIGHT_KEY = "mouthUpperUpRight"
    NOSE_SNEER_LEFT_KEY = "noseSneerLeft"
    NOSE_SNEER_RIGHT_KEY = "noseSneerRight"
    NOSESNEERRIGHT_KEY = "noseSneerRight"
    INV_CHEEK_SQUINT_LEFT_KEY = "INV cheekSquintLeft"
    INV_CHEEK_SQUINT_RIGHT_KEY = "INV cheekSquintRight"
    INV_NOSE_SNEER_LEFT_KEY = "INV noseSneerLeft"
    INV_NOSE_SNEER_RIGHT_KEY = "INV noseSneerRight"
    
    # Regional Blendshape Groups
    EYE_BLENDSHAPES = [
        "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", 
        "browOuterUpRight", "eyeWideLeft", "eyeWideRight"
    ]
    
    MOUTH_BLENDSHAPES = [
        "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "jawOpen", "mouthFunnel",
        "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight"
    ]
    
    NOSE_BLENDSHAPES = [
        "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight",
        "browDownLeft", "browDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
        "eyeSquintLeft", "eyeSquintRight"
    ]
    
    # ============================================================================
    # ACTION UNITS (AU)
    # ============================================================================
    # AU Keys
    AU1_KEY = "AU01- Inner Brow Raiser"
    AU2_KEY = "AU02- Outer Brow Raiser"
    AU4_KEY = "AU04- Brow Lower"
    AU5_KEY = "AU05- Upper Lid Raiser"
    AU6_KEY = "AU06- Cheek Raiser"
    AU7_KEY = "AU07- Lid Tightener"
    AU9_KEY = "AU09- Nose Wrinkler"
    AU10_KEY = "AU10- Upper Lip Raiser"
    AU11_KEY = "AU11- Nasolabial Deepener"
    AU12_KEY = "AU12- Lip Corner Puller"
    AU13_KEY = "AU13- Cheek Puffer"
    AU14_KEY = "AU14- Dimpler"
    AU15_KEY = "AU15- Lip Corner Depressor"
    AU16_KEY = "AU16- Lower Lip Depressor"
    AU17_KEY = "AU17- Chin Raiser"
    AU18_KEY = "AU18- Lip Puckerer"
    AU20_KEY = "AU20- Lip Stretcher"
    AU22_KEY = "AU22- Lip Funneler"
    AU23_KEY = "AU23- Lip Tightener"
    AU24_KEY = "AU24- Lip Pressor"
    AU25_KEY = "AU25- Lips Part"
    AU26_KEY = "AU26- Jaw Drop"
    AU27_KEY = "AU27- Mouth Stretch"
    AU28_KEY = "AU28- Lip Suck"
    AU41_KEY = "AU41- Eye Droop"
    AU42_KEY = "AU42- Eye Slit"
    AU43_KEY = "AU43- Eyes Close"
    AU44_KEY = "AU44- Squint"
    AU45_KEY = "AU45- Blink"
    AU46_KEY = "AU46- Wink"
    AU61_KEY = "AU61- Eye Turn Left"
    AU62_KEY = "AU62- Eye Turn Right"
    AU63_KEY = "AU63- Eye Turn Up"
    AU64_KEY = "AU64- Eye Turn Down"
    
    # Regional AU Groups
    EYE_AUS = [
        "AU01- Inner Brow Raiser", "AU02- Outer Brow Raiser", "AU04- Brow Lower", 
        "AU05- Upper Lid Raiser", "AU07- Lid Tightener", "AU41- Eye Droop", 
        "AU42- Eye Slit", "AU43- Eyes Close", "AU44- Squint", "AU45- Blink", 
        "AU46- Wink", "AU61- Eye Turn Left", "AU62- Eye Turn Right", 
        "AU63- Eye Turn Up", "AU64- Eye Turn Down"
    ]
    
    MOUTH_AUS = [
        "AU16- Lower Lip Depressor", "AU17- Chin Raiser", "AU12- Lip Corner Puller", 
        "AU13- Cheek Puffer", "AU14- Dimpler", "AU15- Lip Corner Depressor",
        "AU10- Upper Lip Raiser", "AU18- Lip Puckerer", "AU20- Lip Stretcher", 
        "AU22- Lip Funneler", "AU23- Lip Tightener", "AU24- Lip Pressor", 
        "AU25- Lips Part", "AU26- Jaw Drop", "AU27- Mouth Stretch", "AU28- Lip Suck"
    ]
    
    NOSE_AUS = ["AU06- Cheek Raiser", "AU11- Nasolabial Deepener", "AU09- Nose Wrinkler"]
    
    # AU Dictionaries
    AU_DICT = {
        AU1_KEY: 0, AU2_KEY: 0, AU4_KEY: 0, AU5_KEY: 0, AU6_KEY: 0, AU7_KEY: 0,
        AU9_KEY: 0, AU10_KEY: 0, AU11_KEY: 0, AU12_KEY: 0, AU13_KEY: 0, AU14_KEY: 0,
        AU15_KEY: 0, AU16_KEY: 0, AU17_KEY: 0, AU18_KEY: 0, AU20_KEY: 0, AU22_KEY: 0,
        AU23_KEY: 0, AU24_KEY: 0, AU25_KEY: 0, AU26_KEY: 0, AU27_KEY: 0, AU28_KEY: 0,
        AU41_KEY: 0, AU42_KEY: 0, AU43_KEY: 0, AU44_KEY: 0, AU45_KEY: 0, AU46_KEY: 0,
        AU61_KEY: 0, AU62_KEY: 0, AU63_KEY: 0, AU64_KEY: 0
    }
    
    AUS = {
        AU1_KEY: (0, 5), AU2_KEY: (0, 5), AU4_KEY: (0, 5), AU5_KEY: (0, 5), 
        AU6_KEY: (0, 5), AU7_KEY: (0, 5), AU9_KEY: (0, 5), AU10_KEY: (0, 5),
        AU11_KEY: (0, 5), AU12_KEY: (0, 5), AU13_KEY: (0, 5), AU14_KEY: (0, 5),
        AU15_KEY: (0, 5), AU16_KEY: (0, 5), AU17_KEY: (0, 5), AU18_KEY: (0, 5),
        AU20_KEY: (0, 5), AU22_KEY: (0, 5), AU23_KEY: (0, 5), AU24_KEY: (0, 5),
        AU25_KEY: (0, 5), AU26_KEY: (0, 5), AU27_KEY: (0, 5), AU28_KEY: (0, 5),
        AU41_KEY: (0, 5), AU42_KEY: (0, 5), AU43_KEY: (0, 5), AU44_KEY: (0, 5),
        AU45_KEY: (0, 5), AU46_KEY: (0, 5), AU61_KEY: (0, 5), AU62_KEY: (0, 5),
        AU63_KEY: (0, 5), AU64_KEY: (0, 5)
    }
    
    # ============================================================================
    # EMOTIONS
    # ============================================================================
    FRAME_KEY = "frame"
    HAPPY_KEY = "Joy"
    SAD_KEY = "Sadness"
    FEAR_KEY = "Fear"
    ANGER_KEY = "Anger"
    SURPRISE_KEY = "Surprise"
    CONTEMPT_KEY = "Contempt"
    DISGUST_KEY = "Disgust"
    
    EMOTIONS = ["Joy", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt"]
    EMOTION_CAHRT_LABELS = ["Emotion", "FRAMES"]
    
    EMOTION_GRAPH = {
        "frame": [], SAD_KEY: [], HAPPY_KEY: [], SURPRISE_KEY: [], 
        FEAR_KEY: [], ANGER_KEY: [], CONTEMPT_KEY: [], DISGUST_KEY: []
    }
    
    # ============================================================================
    # CSV HEADERS & DATA STRUCTURE
    # ============================================================================
    BLENDSHAPE_HEADERS_KEY = [
        "frame", "_neutral", BROW_DOWN_LEFT_KEY, BROW_DOWN_RIGHT_KEY, BROW_INNER_UP_KEY,
        BROW_OUTER_UP_LEFT_KEY, BROW_OUTER_UP_RIGHT_KEY, CHEEK_PUFF_KEY, 
        CHEEK_SQUINT_LEFT_KEY, CHEEK_SQUINT_RIGHT_KEY, EYE_BLINK_LEFT_KEY,
        EYE_BLINK_RIGHT_KEY, EYE_LOOK_DOWN_LEFT_KEY, EYE_LOOK_DOWN_RIGHT_KEY,
        EYE_LOOK_IN_LEFT_KEY, EYE_LOOK_IN_RIGHT_KEY, EYE_LOOK_OUT_LEFT_KEY,
        EYE_LOOK_OUT_RIGHT_KEY, EYE_LOOK_UP_LEFT_KEY, EYE_LOOK_UP_RIGHT_KEY,
        EYE_SQUINT_LEFT_KEY, EYE_SQUINT_RIGHT_KEY, EYE_WIDE_LEFT_KEY,
        EYE_WIDE_RIGHT_KEY, JAW_FORWARD_KEY, JAW_LEFT_KEY, JAW_OPEN_KEY,
        JAW_RIGHT_KEY, MOUTH_CLOSE_KEY, MOUTH_DIMPLE_LEFT_KEY, 
        MOUTH_DIMPLE_RIGHT_KEY, MOUTH_FROWN_LEFT_KEY, MOUTH_FROWN_RIGHT_KEY,
        MOUTH_FUNNEL_KEY, MOUTH_LEFT_KEY, MOUTH_LOWER_DOWN_LEFT_KEY,
        MOUTH_LOWER_DOWN_RIGHT_KEY, MOUTH_PRESS_LEFT_KEY, MOUTH_PRESS_RIGHT_KEY,
        MOUTH_PUCKER_KEY, MOUTH_RIGHT_KEY, MOUTH_ROLL_LOWER_KEY,
        MOUTH_ROLL_UPPER_KEY, MOUTH_SHRUG_LOWER_KEY, MOUTH_SHRUG_UPPER_KEY,
        MOUTH_SMILE_LEFT_KEY, MOUTH_SMILE_RIGHT_KEY, MOUTH_STRETCH_LEFT_KEY,
        MOUTH_STRETCH_RIGHT_KEY, MOUTH_UPPER_UP_LEFT_KEY, MOUTH_UPPER_UP_RIGHT_KEY,
        NOSE_SNEER_LEFT_KEY, NOSE_SNEER_RIGHT_KEY, INV_CHEEK_SQUINT_LEFT_KEY,
        INV_CHEEK_SQUINT_RIGHT_KEY, INV_NOSE_SNEER_LEFT_KEY, INV_NOSE_SNEER_RIGHT_KEY
    ]
    
    CSV_HEADER_KEY = BLENDSHAPE_HEADERS_KEY + [
        # AU keys
        AU1_KEY, AU2_KEY, AU4_KEY, AU5_KEY, AU6_KEY, AU7_KEY, AU9_KEY, AU10_KEY,
        AU11_KEY, AU12_KEY, AU13_KEY, AU14_KEY, AU15_KEY, AU16_KEY, AU17_KEY,
        AU18_KEY, AU20_KEY, AU22_KEY, AU23_KEY, AU24_KEY, AU25_KEY, AU26_KEY,
        AU27_KEY, AU28_KEY, AU41_KEY, AU42_KEY, AU43_KEY, AU44_KEY, AU45_KEY,
        AU46_KEY, AU61_KEY, AU62_KEY, AU63_KEY, AU64_KEY,
        # Emotions
        HAPPY_KEY, SAD_KEY, SURPRISE_KEY, FEAR_KEY, ANGER_KEY, DISGUST_KEY, CONTEMPT_KEY
    ]
    
    # ============================================================================
    # FACIAL LANDMARKS & INDICES
    # ============================================================================
    # Blendshape Indices
    BROW_DOWN_LEFT_IDX = 0
    BROW_DOWN_RIGHT_IDX = 1
    BROW_INNER_UP_IDX = 2
    BROW_OUTER_UP_LEFT_IDX = 3
    BROW_OUTER_UP_RIGHT_IDX = 4
    CHEEK_PUFF_IDX = 5
    CHEEK_SQUINT_LEFT_IDX = 6
    CHEEK_SQUINT_RIGHT_IDX = 7
    EYE_BLINK_LEFT_IDX = 8
    EYE_BLINK_RIGHT_IDX = 9
    EYE_LOOK_DOWN_LEFT_IDX = 10
    EYE_LOOK_DOWN_RIGHT_IDX = 11
    EYE_LOOK_IN_LEFT_IDX = 12
    EYE_LOOK_IN_RIGHT_IDX = 13
    EYE_LOOK_OUT_LEFT_IDX = 14
    EYE_LOOK_OUT_RIGHT_IDX = 15
    EYE_LOOK_UP_LEFT_IDX = 16
    EYE_LOOK_UP_RIGHT_IDX = 17
    EYE_SQUINT_LEFT_IDX = 18
    EYE_SQUINT_RIGHT_IDX = 19
    EYE_WIDE_LEFT_IDX = 20
    EYE_WIDE_RIGHT_IDX = 21
    JAW_FORWARD_IDX = 22
    JAW_LEFT_IDX = 23
    JAW_OPEN_IDX = 24
    JAW_RIGHT_IDX = 25
    MOUTH_CLOSE_IDX = 26
    MOUTH_DIMPLE_LEFT_IDX = 27
    MOUTH_DIMPLE_RIGHT_IDX = 28
    MOUTH_FROWN_LEFT_IDX = 29
    MOUTH_FROWN_RIGHT_IDX = 30
    MOUTH_FUNNEL_IDX = 31
    MOUTH_LEFT_IDX = 32
    MOUTH_LOWER_DOWN_LEFT_IDX = 33
    MOUTH_LOWER_DOWN_RIGHT_IDX = 34
    MOUTH_PRESS_LEFT_IDX = 35
    MOUTH_PRESS_RIGHT_IDX = 36
    MOUTH_PUCKER_IDX = 37
    MOUTH_RIGHT_IDX = 38
    MOUTH_ROLL_LOWER_IDX = 39
    MOUTH_ROLL_UPPER_IDX = 40
    MOUTH_SHRUG_LOWER_IDX = 41
    MOUTH_SHRUG_UPPER_IDX = 42
    MOUTH_SMILE_LEFT_IDX = 43
    MOUTH_SMILE_RIGHT_IDX = 44
    MOUTH_STRETCH_LEFT_IDX = 45
    MOUTH_STRETCH_RIGHT_IDX = 46
    MOUTH_UPPER_UP_LEFT_IDX = 47
    MOUTH_UPPER_UP_RIGHT_IDX = 48
    NOSE_SNEER_LEFT_IDX = 49
    NOSE_SNEER_RIGHT_IDX = 50
    INV_CHEEK_SQUINT_LEFT_IDX = 51
    INV_CHEEK_SQUINT_RIGHT_IDX = 52
    INV_NOSE_SNEER_LEFT_IDX = 53
    INV_NOSE_SNEER_RIGHT_IDX = 54
    
    # Regional Blendshape Index Groups
    EYE_BLENDSHAPES_CHECK = [
        BROW_DOWN_LEFT_IDX, BROW_DOWN_RIGHT_IDX, BROW_INNER_UP_IDX, 
        BROW_OUTER_UP_LEFT_IDX, BROW_OUTER_UP_RIGHT_IDX, EYE_SQUINT_LEFT_IDX, 
        EYE_SQUINT_RIGHT_IDX, EYE_BLINK_LEFT_IDX, EYE_BLINK_RIGHT_IDX, 
        EYE_WIDE_LEFT_IDX, EYE_WIDE_RIGHT_IDX
    ]
    
    MOUTH_BLENDSHAPES_CHECK = [
        JAW_OPEN_IDX, MOUTH_FUNNEL_IDX, MOUTH_LOWER_DOWN_LEFT_IDX,
        MOUTH_LOWER_DOWN_RIGHT_IDX, MOUTH_PRESS_LEFT_IDX, MOUTH_PRESS_RIGHT_IDX,
        MOUTH_PUCKER_IDX, MOUTH_ROLL_LOWER_IDX, MOUTH_ROLL_UPPER_IDX,
        MOUTH_SHRUG_LOWER_IDX, MOUTH_SHRUG_UPPER_IDX, MOUTH_SMILE_LEFT_IDX,
        MOUTH_SMILE_RIGHT_IDX, MOUTH_STRETCH_LEFT_IDX, MOUTH_STRETCH_RIGHT_IDX,
        MOUTH_UPPER_UP_LEFT_IDX, MOUTH_UPPER_UP_RIGHT_IDX
    ]
    
    NOSE_BLENDSHAPES_CHECK = [
        CHEEK_SQUINT_LEFT_IDX, CHEEK_SQUINT_RIGHT_IDX, NOSE_SNEER_LEFT_IDX,
        NOSE_SNEER_RIGHT_IDX, BROW_DOWN_LEFT_IDX, BROW_DOWN_RIGHT_IDX,
        MOUTH_UPPER_UP_LEFT_IDX, MOUTH_UPPER_UP_RIGHT_IDX, EYE_SQUINT_LEFT_IDX,
        EYE_SQUINT_RIGHT_IDX
    ]
    
    # Facial Landmark Points
    RIGHT_EYE_POINTS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    LEFT_EYE_POINTS = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 380, 374, 380, 381, 382]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
    OUTER_MOUTH_IDXS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    LEFT_INNER_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    RIGHT_INNER_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    RIGHT_EYE_IRIS = 468
    LEFT_EYE_IRIS = 473
    
    # ============================================================================
    # FEATURE INDICES - CRAFTED FEATURES
    # ============================================================================
    # Mouth Features
    LIP_OUTER_H_DST_IDX = 0
    LIP_OUTER_V_DST_IDX = 1
    LIP_INNER_V_DST_IDX = 2
    MOUTH_CEN_CHIN_DST_IDX = 3
    UPPER_LIP_NOSE_TIP_DST_IDX = 4
    LIP_UPPER_ROLL_HEIGHT_IDX = 5
    LIP_LOWER_ROLL_HEIGHT_IDX = 6
    LEFT_LIP_V_IDX = 7
    RIGHT_LIP_V_IDX = 8
    MOUTH_OUT_V_IDX = 9
    NOSE_TIP_CHIN_DST_IDX = 10
    LIP_INNER_H_DST_IDX = 11
    LIP_TIGHT_D1_IDX = 12
    LIP_TIGHT_D2_IDX = 13
    LIP_TIGHT_D3_IDX = 14
    LIP_TIGHT_D4_IDX = 15
    LIP_U_OUT_NOSE_TIP_DST_IDX = 16
    LIP_ROLL_HEIGHT_IDX = 17
    LIP_DOWN_V_DST_IDX = 18
    FACE_WIDTH_IDX = 19
    FACE_HEIGHT_IDX = 20
    
    # Nose Features
    RIGHT_NOSE_BRIDGE_TIP_MAX_DST_IDX = 21
    RIGHT_NOSE_BRIDGE_TIP_MIN_DST_IDX = 22
    LEFT_NOSE_BRIDGE_TIP_MAX_DST_IDX = 23
    LEFT_NOSE_BRIDGE_TIP_MIN_DST_IDX = 24
    RIGHT_NASOLABIAL_MAX_DST_IDX = 25
    RIGHT_NASOLABIAL_MIN_DST_IDX = 26
    LEFT_NASOLABIAL_MAX_DST_IDX = 27
    LEFT_NASOLABIAL_MIN_DST_IDX = 28
    LEFT_CHEEK_RAISE_DST_IDX = 29
    RIGHT_CHEEK_RAISE_DST_IDX = 30
    LEFT_CHEEK_RAISE_UP_DST_IDX = 31
    RIGHT_CHEEK_RAISE_UP_DST_IDX = 32
    
    # Eye Features
    LEFT_EYE_WIDE_IDX = 33
    RIGHT_EYE_WIDE_IDX = 34
    LEFT_EYE_INNER_DST_IDX = 35
    RIGHT_EYE_INNER_DST_IDX = 36
    EYE_WIDTH_INDEX = 37
    LEFT_BROW_LOWER_IDX = 38
    RIGHT_BROW_LOWER_IDX = 39
    LEFT_BROW_INNER_RAISER = 40
    RIGHT_BROW_INNER_RAISER = 41
    LEFT_BROW_OUTER_RAISER = 42
    RIGHT_BROW_OUTER_RAISER = 43
    LEFT_BROW_CEN_DST_IDX = 44
    RIGHT_BROW_CEN_DST_IDX = 45
    
    # Extreme Values Indices
    LIP_H_STRETCH_IDX = 0
    LIP_V_STRETCH_IDX = 1
    LIP_FROWN_PULL_MAX_IDX = 2
    LIPS_PART_IDX = 3
    JAW_DROP_IDX = 4
    MOUTH_STRETCH_V_IDX = 5
    UPPER_LIP_RAISE_IDX = 6
    START_FPT_WIDTH_IDX = 7
    EXTREME_FPT_WIDTH_IDX = 8
    MOUTH_PRESS_MAX_IDX = 14
    MOUTH_PRESS_MIN_IDX = 15
    MOUTH_CLOSE_JAW_DROP_MAX_IDX = 16
    LIP_CORNER_DEPRESS_MAX_IDX = 17
    LEFT_UPPER_LID_MAX_IDX = 18
    RIGHT_UPPER_LID_MAX_IDX = 19
    LEFT_LID_TIGHT_MAX_IDX = 20
    RIGHT_LID_TIGHT_MAX_IDX = 21
    LEFT_INNER_BROW_MAX_IDX = 22
    RIGHT_INNER_BROW_MAX_IDX = 23
    LEFT_BROW_LOWER_MAX_IDX = 24
    RIGHT_BROW_LOWER_MAX_IDX = 25
    LEFT_OUTER_BROW_MAX_IDX = 26
    RIGHT_OUTER_BROW_MAX_IDX = 27
    LEFT_EYE_DROOP_IDX = 28
    RIGHT_EYE_DROOP_IDX = 29
    LEFT_EYE_SLIT_IDX = 30
    RIGHT_EYE_SLIT_IDX = 31
    LEFT_EYES_QUINT_IDX = 32
    RIGHT_EYES_QUINT_IDX = 33
    
    NOSE_FEATURES_START_IDX = 21
    EYE_FEATURES_START_IDX = 33
    
    # ============================================================================
    # THRESHOLDS - ACTION UNIT DETECTION
    # ============================================================================
    # Eye Thresholds
    AU7_THRESH_LEFT = [0.1, 0.3]
    AU7_THRESH_RIGHT = [0.1, 0.3]
    WINK_LEFT_THRESH = [0.3, 0.25]
    WINK_RIGHT_THRESH = [0.25, 0.20]
    BLINK_THRESH_LEFT = 0.35
    BLINK_THRESH_RIGHT = 0.35
    CLOSE_FRAME_THRESH = 10
    SQUINT_LEFT_THRESH = 0.5
    SQUINT_RIGHT_THRESH = 0.5
    DROOP_THRESH_LEFT = 0.004
    DROOP_THRESH_RIGHT = 0.004
    SLIT_THRESH_LEFT = 0.0025
    SLIT_THRESH_RIGHT = 0.0025
    AU5_THRESH_LEFT = 0.3
    AU5_THRESH_RIGHT = 0.3
    
    # Brow Thresholds
    AU4_THRESH_LEFT = [0.2, 0.4, 0.6, 0.8]
    AU4_THRESH_RIGHT = [0.2, 0.4, 0.6, 0.8]
    AU1_THRESH = [0.25, 0.4, 0.6, 0.8]
    AU2_THRESH_LEFT = [0.2, 0.4, 0.6, 0.8]
    AU2_THRESH_RIGHT = [0.2, 0.4, 0.6, 0.8]
    
    # Mouth Thresholds
    AU16_LEFT_THRESH = 0.001
    AU16_RIGHT_THRESH = 0.001
    AU22_THRESH = 0.05
    AU18_THRESH = [0.15, 0.30, 0.50, 0.7]
    AU13_THRESH = 3e-5
    AU14_THRESH_LEFT = 0.02
    AU14_THRESH_RIGHT = 0.02
    AU24_THRESH_LEFT = 0.05
    AU24_THRESH_RIGHT = 0.05
    AU20_THRESH = 0.40
    AU27_THRESH_LEFT = 0.01
    AU27_THRESH_RIGHT = 0.01
    AU26_THRESH = [0.15, 0.30, 0.50, 0.70]
    AU10_THRESH = 0.01
    AU12_THRESH_LEFT = 0.1
    AU12_THRESH_RIGHT = 0.1
    AU15_THRESH_LEFT = 0.01
    AU15_THRESH_RIGHT = 0.01
    AU28_THRESH_LEFT = 0.1
    AU28_THRESH_RIGHT = 0.1
    AU23_THRESH_LEFT = 0.001
    AU23_THRESH_RIGHT = 0.001
    AU25_JAW_THRESH = [0.008, 0.01, 0.04, 0.1, 0.15]
    
    # Nose/Cheek Thresholds
    AU6_THRESH_LEFT = [0.001, 0.1]
    AU6_THRESH_RIGHT = [0.001, 0.1]
    AU9_THRESH_LEFT = [0.02, 0.1, 0.2, 0.4]
    AU9_THRESH_RIGHT = [0.02, 0.1, 0.2, 0.4]
    
    # Chin Thresholds
    AU17_MOUTH_ROLL_THRESH = [0.05, 0.1, 0.15, 0.2]
    AU17_MOUTH_SHRUG_THRESH = [0.2, 0.35, 0.55, 0.75]
    
    # Eye Look Direction Thresholds
    LOOK_LEFT_THRESH = 0.25
    LOOK_RIGHT_THRESH = 0.25
    LOOK_UP_THRESH = 0.05
    LOOK_DOWN_THRESH = 0.05
    
    # Blendshape Thresholds
    EYE_BLENDSHAPE_THRESHOLD = 0.25
    MOUTH_BLENDSHAPES_THRESHOLD = 0.01
    NOSE_BLENDSHAPE_THRESHOLD = 0.25
    
    # Weight Factors
    AU6_MOUTH_WEIGHT = 0.5
    AU7_WEIGHT = 0.5
    MOUTH_SHRUG_UP_WEIGHT = 0.7
    BROW_LOWER_WEIGHT = 0.5
    AU17_MOUTH_ROLL_WEIGHT = 0.2
    AU17_MOUTH_SHRUG_WEIGHT = 0.8
    
    # ============================================================================
    # STATE MACHINE & ACTION TRACKING
    # ============================================================================
    QUEUE_LENGTH = 7
    MAX_QUEUE_LEN = 15
    SMALL_VAL_BLENDSHAPES = [6, 7, 49, 50]
    
    JUMP_UNI_STATE = "Uni"
    JUMP_INC_STATE = "Inc"
    JUMP_DEC_STATE = "Dec"
    
    COUNT_FIVE = 5
    COUNT_TWO = 2
    COUNT_ONE = 1
    COUNT_ZERO = 0
    
    START = 'start'
    STOP = "stop"
    ACTION_START = "Action Start"
    ACTION_END = "Action End"
    PROGRESS = "Progress"
    REDUCE = "Reduce"
    HOLD = "Hold"
    MAXIMA = "Maxima"
    MINIMA = "Minima"
    REST = "Rest"
    
    ACTION_STATE = {
        START: 1,
        STOP: 0,
        PROGRESS: -1,
        REDUCE: -1,
        HOLD: -1,
        REST: 0
    }
    
    INV_KEYS = {6: 51, 7: 52, 49: 53, 50: 54}
    INV_DIVISORS = {6: 0, 7: 1, 49: 2, 50: 3}
    
    # ============================================================================
    # AUDIO FEATURE EXTRACTION
    # ============================================================================
    # Audio Feature Configuration
    RF_TOP_K = 5
    TOP_K = 5
    RANDOM_STATE = 42
    N_ESTIMATORS = 200
    
    # Cross-Validation Settings
    MIN_CV_FOLDS = 2
    MAX_CV_FOLDS = 5
    CV_DIVISOR = 10
    CV_SCORING = "neg_mean_squared_error"
    
    # Data Cleaning
    MIN_ROWS_REQUIRED = 5
    DROP_NA_AXIS = 0
    DROP_NA_HOW = "any"
    
    # Target Column & Feature Filtering
    TARGET_COLUMN = "rms"
    FRAME_IDX_KEY = "frame_index"
    NON_FEATURE_COLS = {
        "frame_index",
        "frame_start_s",
        "frame_end_s",
        "frame_seconds",
        "hop_seconds",
        "rms",
        "frame",
        "timestamp",
        "file",
        "label",
    }
    
    # Audio Processing Parameters
    DEFAULT_FRAME_SECONDS = 1.0
    DEFAULT_HOP_RATIO = 0.5
    DEFAULT_WINDOWS = [1.0, 0.5, 0.1, 0.05, 0.025]
    LIBROSA_MONO = True
    N_MFCC = 13
    F0_MIN_HZ = 50
    
    # Feature Column Names
    FEATURE_FRAME_INDEX = "frame_index"
    FEATURE_START_S = "frame_start_s"
    FEATURE_END_S = "frame_end_s"
    FEATURE_RMS = "rms"
    FEATURE_SPECTRAL_CENTROID = "spectral_centroid"
    FEATURE_SPECTRAL_BANDWIDTH = "spectral_bandwidth"
    FEATURE_ZCR = "zcr"
    FEATURE_FRAME_SECONDS_COL = "frame_seconds"
    FEATURE_HOP_SECONDS = "hop_seconds"
    FEATURE_F0_HZ = "f0_hz"
    FEATURE_MFCC_PREFIX = "mfcc_"
    FEATURE_CHROMA_PREFIX = "chroma_"
    
    # Audio Analysis Keys
    WINDOW_LEN_S_KEY = "window_length_s"
    TOP_RF_FEATURES_KEY = "top_rf_features"
    ACOUSTIC_CLAIM_KEY = "acoustic_claim"
    BODY_KEY = "body"
    NOT_ASSIGN_KEY = "N/A"
    SPEAKER_KEY = "speaker"
    CONCLUSION_KEY = "conclusions"
    EVIDENCE_KEY = "evidence"
    UNKNOWN_KEY = "Unknown"
    
    # ============================================================================
    # FILE HANDLING & JSON
    # ============================================================================
    FILE_ENCODING = "utf-8"
    UTF_8 = 'utf-8'
    JSON_EXT = ".json"
    TXT_EXTENSION = ".txt"
    FORCE_ASCII = False
    JSON_INDENT = 2
    
    # JSON Preview
    DEFAULT_PREVIEW_FRAMES = 6
    PREVIEW_N_FRAMES = 6
    TRUNCATED_TEXT_MAX_LEN = 2000
    
    # File Patterns
    BATCH_NAME_PATTERN = r"(batch_\d+)"
    MARKUP_CLEANUP_PATTERN = r"```json\s*(\{.*?\})\s*```"
    JSON_PATTERN_1 = r'```(?:json)?\s*(\{.*?\})\s*```'
    FALLBACK_JSON_PATTERN = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    # ============================================================================
    # DIRECTORY & FILE STRUCTURE
    # ============================================================================
    # Base Directories
    CURRENT_DIR = "."
    WORK_DIR = "Output"
    VISION_OUT_DIR = f"{WORK_DIR}/Vision/"
    AUDIO_OUT_DIR = f"{WORK_DIR}/Audio/"
    
    # Pipeline Output Structure
    # BASE_OUTPUT_DIR = Path(os.getenv("PIPELINE_OUTPUT_DIR", "./pipeline_output"))
    DECODED_FILES_DIR = f"{WORK_DIR}/{MULTIMODAL_STR}/decoded_audio_and_vision_csv"
    BATCH_INPUTS_DIR = f"{WORK_DIR}/{MULTIMODAL_STR}/batch_inputs"
    BATCH_OUTPUTS_DIR = f"{WORK_DIR}/{MULTIMODAL_STR}/batched_outputs"
    AUDIO_FEATURES_DIR = f"{BATCH_OUTPUTS_DIR}/audio_outputs"
    VISION_FEATURES_DIR = f"{BATCH_OUTPUTS_DIR}/vision_outputs"
    MULTIMODAL_RESULTS_DIR = f"{BATCH_OUTPUTS_DIR}/multimodal_outputs"

    
    # Folder Names
    VISION_OUTPUT_FOLDER_NAME = "vision_outputs"
    AUDIO_OUTPUTS_FOLDER_NAME = "audio_outputs"
    EXTRACTED_FILES_FOLDER = "extracted_files"
    
    # File Suffixes
    PREFILTER_FRAMES_CSV_SUFFIX = "_prefiltered_frames.csv"
    BLENDSAHPE_AU_EMOT_CSV_SUFFIX = "_blendshapes_AU_emotions_filtered.csv"
    SAMPELD_BLENDSAHPE_CSV_SUFFIX = "_sampled_blendshape.csv"
    EXTRACTED_AUDIO_SUFFIX = "_extracted_audio.mp3"
    RESULT_JSON_SUFFIX = "result.json"
    
    # Standard Filenames
    PREFILTER_FRAME_CSV = "prefiltered_frames.csv"
    SAMPELED_BLENDSAHPE_CSV = "sampled_blendshape.csv"
    BLENDSHAPE_AU_EMOTION_FILTERED_KEY = "blendshapes_AU_emotions_filtered.csv"
    VISION_BATCH_CSV = "vision_batch.csv"
    VISION_FEATURES_CSV = "vision_features.csv"
    AUDIO_WAV_FILE = "audio_batch.wav"
    FINAL_REPORT_TXT = "final_summary_report.txt"
    BATCH_AUDIO = "batch_audio"
    
    # ============================================================================
    # SESSION STATE & PROCESSING KEYS
    # ============================================================================
    # Processing State Keys
    FRAME_NO_KEY = "frame_no"
    CSV_PATH_KEY = "csv_path"
    REGION_MAP_KEY = "region_map"
    AU_REGION_MAP_KEY = "au_region_map"
    EMOTION_COLS_KEY = "emotion_cols"
    OUT_DIR_KEY = "out_dir"
    FILTERED_CSV_PATH_KEY = "filtered_csv_path"
    FILTERED_CSV_RESULT_KEY = "filtered_csv_result"
    PREFILTER_RESULT_KEY = "prefilter_result"
    ORIGINAL_FPS_KEY = "original_fps"
    SAMPELED_CSV_PATH_KEY = "sampled_csv_path"
    CSV_SAMPLER_RESULT_KEY = "csv_sampler_result"
    CSV_SAMPLER_KEY = "csv_sampler"
    PREFILTER_DECISION_KEY = "prefilter_decision"
    ACTIVE_REGION_KEY = "active_regions"
    ORCHESTRATOR_KEY = "orchestrator_decision"
    ORCHESTRATOR_MEMORY_SUMMARY_KEY = "orchestrator_memory_summary"
    NEXT_TOOL_KEY = "next_tool"
    
    # Result Keys
    TARGET_FPS_KEY = "target_fps"
    SUCCESS_KEY = "success"
    REASON_KEY = "reason"
    REASONING_KEY = "reasoning"
    REGIONS_KEY = "regions"
    USEFUL_KEY = "useful"
    IS_VALID_KEY = "is_valid"
    HAS_DATA_KEY = "has_data"
    IS_MEDICAL_KEY = "is_medical"
    IS_ALIGNED_KEY = "is_aligned"
    DATA_ROW_COUNT_KEY = "data_row_count"
    ERROR_KEY = "error"
    NAME_KEY = 'name'
    SUMMARY_KEY = "summary"
    EVIDENCE_LIST_KEY = "evidence_list"
    INTERPRETATION_KEY = "interpretation"
    FRAME_RANGE_KEY = "frame_range"
    FRAME_INDEX_KEY = "frame_index"
    MODALITY_KEY = "modaltiy"
    DISEASE_FOCUS_KEY = "disease_focus"
    PREFILTER_SUMMARY_KEY = "prefilter_summary"
    WORK_DIR_KEY = "work_dir"
    BATCH_LABEL_KEY = "batch_label"
    BATCH_RESULTS_KEY = "batch_results"
    BATCHES = "batches"
    
    # Meta Intent Keys
    METAINTENT_RES_KEY = "meta_intent_result"
    METAINTENT_ANALYSIS_KEY = "meta_intent_analysis"
    INTENT_TYPE = "intent_type"
    
    # Summary Keys
    PREFILTER_KEY = "prefilter"
    CSV_FILTER_KEY = "csv_filter"
    FILTERED_BLENDSHAPE_CSV_PATH_KEY = "filtered_blendshape_csv_path"
    SYMPTOM_ANALYSIS_KEY = "symptom_analysis"
    BLENDSHAPE_CSV_PATH_KEY = "blendshape_csv_path"
    PREFILTER_CSV_PATH_KEY = "prefiltered_csv_path"
    SAMPELED_BLENDSAHPE_CSV_PATH_KEY = "sampled_blendshape_csv_path"
    KEPT_RANGES_KEY = "kept_ranges"
    
    # ============================================================================
    # UI DISPLAY KEYS
    # ============================================================================
    # Results Display
    FINAL_SUMMERY_KEY = "Final Summary"
    BLENDSHAPE_KEY = 'blendshapes'
    FEATURES_KEY = "Features"
    AVG_ACTIVATION_KEY = "Avg Activation"
    MAX_ACTIVATION_KEY = "Max Activation"
    LINE_MODE_KEY = 'lines'
    ACTIVATION_KEY = "Activation"
    TOP_KEY = "top"
    LEFT_KEY = "left"
    RIGHT_KEY = "right"
    DOWN_KEY = "down"
    HOVERMODE_KEY = 'x unified'
    
    # Chart Labels
    BLENDSHAPES_LAYOUT_HEADING = "Region Blendshapes Over Time"
    TOP_ACTIVATED_BLEND_HEADING = "**Top 5 Most Activated Features:**"
    NO_BLENSHAPE_KEY = "blendshapes found in data"
    NO_KEY = "No"
    COLOR_RANGES = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    EMOTION_LAYOUT_KEY = "Emotions Over Time"
    EMOTION_Y_LABEL = "Emotion Score"
    EMOTION_KEY = "emotion"
    TIMESERIES_KEY = "timeseries"
    TIMESTAMP_KEY = "timestamp"
    MESSAGE_KEY = "message_key"
    TOP_ACTIVATED_EMOTION_HEADING = "**Top 5 Most Activated Emotions:**"
    NO_EMOTION_DATA = "No emotion data found in data."
    
    # Pain Display
    PAIN_COLS_KEY = 'pain_cols'
    PAIN_HEADING = "### 😣 Pain Over Time"
    PAIN_LAYOUT_HEADING = "Pain Over Time"
    PAIN_SCORE_KEY = "Pain Score"
    PAIN_COL_COLUMN = "Pain columns found but no 'frame' column."
    NO_PAIN_DATA = "No pain data found in data."
    PAIN_LAYOUT_HEIGHT = 250
    
    # AU Display
    AUS_KEY = 'aus'
    AUS_FOUND_KEY = "AUs Found"
    AU_INTENSITY_KEY = "AU Intensity"
    AU_LAYOUT_TITLE = "Region Action Units Over Time"
    AU_HEATMAP_HEADING = "**AU Activation Heatmap:**"
    COLOR_BAR_TITLE_KEY = "Intensity"
    COLOR_SCALE_KEY = 'Viridis'
    ACTION_UNIT = "Action Unit"
    AU_ACTIVATION_TITLE = "AUs Activation Heatmap"
    AU_HEATMAP_STR = "_au_heatmap_"
    TOP_AUS_ACTIVATED_HEADING = "**Top 5 Most Activated AUs:**"
    NO_AU_FOUND = "Action Units found in data"
    
    # Chart Headings
    BLENDSHAPE_HEADING = "Mediapipe Blendshapes"
    EMOTION_HEADING = "Emotion Detected"
    EMOTION_CHART_HEADING = "Emotion Charts"
    BLENDHSAPE_TABLE_HEADING = ["Blendshape", "value"]
    DECISON_FRAME = "Decison Frame"
    DECISOON_FACE_FRAME = "Decision Face Frame"
    AU_INTENSITY = "AU Intensity"
    DECISION_FRAME_MESH = "Decision Frame Mesh"
    BLENDSHAPE_VALS = "Blendshape Vals"
    ROTATION_VALS = "Rotation Vals"
    EMOTION_DETECTION = "Emotion Detection"
    EMOTION_GRAPH_DATA = "Emotion Graph Data"
    
    # ============================================================================
    # UI BUTTONS & CONTROLS
    # ============================================================================
    START_ANAYSIS_KEY = "🚀 Start Analysis"
    BUTTON_TYPE_KEY = "primary"
    RESET_BTN_KEY = "🔄 Reset"
    STATUS_KEY = 'status'
    COMPLETED_KEY = 'Completed'
    PENDING_KEY = 'Pending'
    
    # ============================================================================
    # REGIONAL KEYS
    # ============================================================================
    EYES_KEY = "eyes"
    MOUTH_KEY = "mouth"
    NOSE_KEY = "nose"
    
    # ============================================================================
    # NUMERIC CONSTANTS
    # ============================================================================
    # Common Numbers
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    TEN = 10
    THIRTY = 30
    NINTY = 90
    HUNDERD = 100
    ONE_EIGHTY = 180
    THREE_HUNDERD = 300
    FIVE_HUNDERD = 500
    THOUSAND = 1000
    
    # Decimal Constants
    EPSILON = 10e-7
    POINT_ZERO_ONE = 0.01
    POINT_ONE = 0.1
    POINT_TWO = 0.2
    POINT_TWO_FIVE = 0.25
    POINT_THREE = 0.3
    POINT_FOUR = 0.4
    HALF = 0.5
    POINT_SEVEN_FIVE = 0.75
    POINT_EIGHT = 0.8
    ONE_POINT_ONE = 1.1
    ONE_TWO_FIVE = 1.25
    TWO_THIRD = 2/3
    ONE_THIRD = 1/3
    TEN_POW_SEVEN = 0.0000001
    CONSTANT_ONE = 1.0
    CONSTANT_MS = 1000
    
    # Display Configuration
    FEATURE_LIMITS = 15
    HEIGHT = 400
    NBINS = 20
    X = 1.01
    Y = 0.99
    SAMPLE_LIMIT = 50
    CHART_MAX_FRAME = 200
    
    # ============================================================================
    # STRING CONSTANTS
    # ============================================================================
    # Common Strings
    UNDERSCORE = "_"
    DOT = "."
    COMMA = ","
    NEWLINE = "\n"
    INVERTED_STRING = ""
    EMPTY_STRING = ""
    
    # File Modes
    READ_BINARY = "rb"
    WRITE_BINARY = "wb"
    READ_MODE = "r"
    READ_MDOE = 'r'
    WRITE_MODE = "w"
    APPEND_MODE = "a"
    
    # Role Keys
    USER_CONTENT = "user_content"
    ASSISTENT_KEY = "assistant"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    
    # Colors
    WHITE_COLOR = '#ffffff'
    WAVE_COLOR = "#0ea5a4"
    RED = (255, 0, 0)
    GREEN = (0, 255, 255)
    YELLOW = (255, 255, 0)
    
    # ============================================================================
    # CONFIGURATION & ENVIRONMENT
    # ============================================================================
    # API Configuration
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    ENV_FILE_PATH = "./env"
    
    # Processing Configuration
    DEFAULT_FPS = 30
    FILE_SIZE_LIMIT = 100  # MB
    VISION_DEFAULT_CSV_READ_NUM = 5000
    RUNNING_STATE = 'is_running'
    FRAME_COUNTS = "frame counts"
    
    # ============================================================================
    # ORCHESTRATOR & MANIFEST
    # ============================================================================
    AUDIO_MANIFEST_HEADER = "Audio Feature Manifest (Batch: {batch_label}):\n"
    PREVIEW_PREFIX = "PREVIEW audio_window_s={w:.3f}: "
    AUDIO_FILENAME_PREFIX = "AUDIO_FILENAME: "
    RF_GUIDANCE_TEXT_TEMPLATE = "RF_GUIDANCE (window_s={w:.3f}): {rf_json}"
    
    # ============================================================================
    # ERROR MESSAGES & WARNINGS
    # ============================================================================
    UNSUPPORTED_FORMAT = "Unsupported file format"
    WAVEFPORM_ERROR = "Could not plot waveform:"
    METAINTENT_NO_RESULT_STR = "⚠️ No meta-intent result returned by LLM."
    METAINTENT_NO_ANALYSIS_STR = "⚠️ No user message found for MetaIntent analysis."
    CSV_NOT_FOUND = "CSV not found"
    
    # ============================================================================
    # COMMAND LINE ARGUMENTS
    # ============================================================================
    PROMPT_KEY = 'prompt'
    VIDEO_PATH_KEY = "video_path"
    AUDIO_PATH_KEY = "audio_path"
    VISION_CSV_PATH_KEY = "vision_csv_path"
    BATCH_SIZE = "batch_size"
    MODEL_NAME_KEY = "model_name"
    INPUT_FPS_KEY = "input_fps"