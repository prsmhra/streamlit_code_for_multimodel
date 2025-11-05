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
