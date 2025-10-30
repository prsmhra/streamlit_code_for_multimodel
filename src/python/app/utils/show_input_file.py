import librosa
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from pydub import AudioSegment

from src.python.app.constants.constants import Constants

def show_audio_waveform(audio_file, waveform_container, audio_player_container):
    # try:
    #     y, sr = librosa.load(audio_file, sr=None)

    #     duration = librosa.get_duration(y=y, sr=sr)
    #     st.session_state.audio_duration = duration
    #     st.session_state.audio_data = (y, sr)
        
    #     fig, ax = plt.subplots(figsize=Constants.AUDIO_FRAME_SIZE)
    #     librosa.display.waveshow(y, sr=sr, ax=ax, color=Constants.WAVE_COLOR)
    #     ax.set_xlabel(Constants.X_LABEL, fontsize=10)
    #     ax.set_ylabel(Constants.Y_LABEL, fontsize=10)
    #     ax.set_facecolor(Constants.WHITE_COLOR)
    #     fig.patch.set_facecolor(Constants.WHITE_COLOR)
    #     plt.tight_layout()
    #     waveform_container.pyplot(fig)
        
    #     # with open(audio_file, Constants.READ_BINARY) as audio_f:
    #     audio_bytes = audio_file.read()
    #     audio_player_container.audio(audio_bytes, format=Constants.AUDIO_FORMAT)
    # except Exception as e:
    #     waveform_container.error(f"{Constants.WAVEFPORM_ERROR} {e}")


    
    # try:
    #     # Save uploaded file to a temporary location
    #     import tempfile
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
    #         tmp.write(audio_file.read())
    #         tmp_path = tmp.name

    #     # Load audio using pydub
    #     audio = AudioSegment.from_file(tmp_path)
    #     samples = np.array(audio.get_array_of_samples())
    #     sr = audio.frame_rate
    #     duration = len(samples) / sr
    #     st.session_state.audio_duration = duration
    #     st.session_state.audio_data = (samples, sr)

    #     # Plot waveform
    #     fig, ax = plt.subplots(figsize=Constants.AUDIO_FRAME_SIZE)
    #     ax.plot(np.linspace(0, duration, num=len(samples)), samples, color=Constants.WAVE_COLOR)
    #     ax.set_xlabel(Constants.X_LABEL, fontsize=10)
    #     ax.set_ylabel(Constants.Y_LABEL, fontsize=10)
    #     ax.set_facecolor(Constants.WHITE_COLOR)
    #     fig.patch.set_facecolor(Constants.WHITE_COLOR)
    #     plt.tight_layout()
    #     waveform_container.pyplot(fig)

    #     # Play audio
    #     audio_file.seek(0)
    #     audio_bytes = audio_file.read()
    #     audio_player_container.audio(audio_bytes, format=Constants.AUDIO_FORMAT)

    # except Exception as e:
    #     waveform_container.error(f"{Constants.WAVEFPORM_ERROR} {e}")


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
