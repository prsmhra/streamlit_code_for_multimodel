# Multimodal Medical Agent System

An AI-powered medical analysis application for video, Audio or Multimodel, analysis of various medical examination using agentic frame using Gemini AI model.

## Table of contents
- Features
- Repository layout
- Requirements
- Quickstart


## Features
- Agent abstraction layer for pluggable models (local and remote)
- Workflow composition (sequential, parallel, conditional)
- Logging and basic metrics
- Extensible adapters for new model providers

## Repository layout

     .
     ├── assets
     │   ├── abbott.png
     │   ├── efficientdet_lite0_8.tflite
     │   └── face_landmarker.task
     ├── Config
     │   ├── config.py
     │   └── logger_setup.py
     ├── main.py
     ├── README.md
     ├── requirement.txt
     ├── requirements.txt
     └──src
        └── python
            └── app
                ├── common
                │   ├── audio_agent_integration.py
                │   ├── cli.py
                │   ├── extractor.py
                │   ├── gemini_client.py
                │   ├── inference.py
                │   ├── orchestrator.py
                │   ├── rf_analysis.py
                │   ├── summary.py
                │   ├── uploader.py
                │   ├── vision_agent_call_old.py
                │   ├── vision_agent_call.py
                │   ├── vision_agents.py
                │   ├── vision_single_batch_runner.py
                │   └── web_ui.py
                ├── constants
                │   └── constants.py
                ├── instructions
                │   ├── audio_agent_instructions.py
                │   └── vision_agent_instructions.py
                ├── tools
                │   ├── csv_filter_tools.py
                │   ├── frame_prefilter_tools.py
                │   └── sample_function_tools.py
                ├── utils
                │   ├── agents_logs.py
                │   ├── batching.py
                │   ├── data_utils.py
                │   ├── draw_meshpoints.py
                │   ├── extract_josn_from_text.py
                │   ├── get_list_from_str.py
                │   ├── show_input_file.py
                │   ├── state_summary.py
                │   ├── thread_pool_executer.py
                │   └── ui_renders.py
                └── video_frame_extractor
                    ├── au_detection.py
                    ├── csv_sav_inference.py
                    ├── detect_emotion.py
                    ├── detect_pain.py
                    ├── face_detector.py
                    ├── get_crafted_features.py
                    ├── image_crop_align.py
                    ├── process_blendshape_detection.py
                    ├── queueExecution.py
                    └── video_stabilizer.py


     3 directories, 49 files


     

## Requirements
- Language runtime: Python 3.10+ 
- GEMINI api key (set via env)

1. Create and activate a virtual environment 
     ```
     python -m venv venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```


2. Run the app locally:
     ```
     streamlit rum main.py
     ```




<!-- ## License
Include the appropriate LICENSE file in the repository (e.g., MIT, Apache 2.0). Update this section to reflect the chosen license. -->

<!-- ## Support / Contact
Open an issue in the repository for bugs or feature requests.

Feel free to adjust sections above to match your actual implementation details and runtime choices. 1Q2345   -->