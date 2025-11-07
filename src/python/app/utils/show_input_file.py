import librosa
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import subprocess  
from pathlib import Path  
from pydub import AudioSegment

from src.python.app.constants.constants import Constants

def show_audio_waveform(audio_file, waveform_container, audio_player_container):
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Try loading with pydub
        audio = AudioSegment.from_file(tmp_path)
        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        duration = len(samples) / sr

        # Plot waveform
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.linspace(0, duration, num=len(samples)), samples)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # Play audio
        audio_file.seek(0)
        st.audio(audio_file, format="audio/mp3")

    except Exception as e:
        st.error(f"Could not plot waveform: {e}")



def show_csv_files(file_name, uploaded_file, container):
    df = None
    if file_name.endswith(f"{Constants.DOT}{Constants.CSV_EXT[Constants.ZERO]}"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(f"{Constants.DOT}{Constants.CSV_EXT[Constants.ONE]}"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error(Constants.UNSUPPORTED_FORMAT)

    container.dataframe(df)
    return df









def show_video_preview(video_file, container):
    """
    Displays a video preview. If the format is not web-safe (like .avi or .mkv),
    it converts it to a temporary .mp4 file for display.
    """
    file_name = video_file.name
    file_ext = Path(file_name).suffix.lower()
    
    # These formats can be played directly by the browser
    web_safe_formats = ['.mp4', '.webm', '.ogv']

    if file_ext in web_safe_formats:
        container.video(video_file)
        return

    # --- If not web-safe, we must convert it ---
    container.info(f"ℹ️ Converting {file_ext} to .mp4 for preview... This may take a moment.")
    
    input_path = None
    output_path = None
    
    try:
        # 1. Save the uploaded file to a temp file
        video_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_in:
            tmp_in.write(video_file.read())
            input_path = tmp_in.name
            
        # 2. Create a path for the output .mp4 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
            output_path = tmp_out.name

        # 3. Run ffmpeg to convert the file
        # -i: input file
        # -c:v copy: (fastest) copy video stream without re-encoding
        # -c:a aac: (fast) re-encode audio to AAC, which is browser-safe
        # -y: overwrite output file
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "copy", 
            "-c:a", "aac",
            "-y", output_path
        ]
        
        # Run the command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 4. Display the new .mp4 file
        if Path(output_path).exists():
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                container.video(video_bytes)
        else:
            raise RuntimeError("ffmpeg conversion failed silently.")

    except Exception as e:
        container.error(f"Could not display video preview: {e}")
        container.info("Make sure 'ffmpeg' is installed and accessible in your system's PATH.")
        
    finally:
        # 5. Clean up both temp files
        if input_path and Path(input_path).exists():
            os.unlink(input_path)
        if output_path and Path(output_path).exists():
            os.unlink(output_path)