"""
Extracts frame-level acoustic features using librosa and returns them as a pandas DataFrame.

This is adapted from your Colab implementation (extract_features_to_df_param),
used in the multi-window Gemini pipeline. It computes:
- RMS energy
- Spectral centroid / bandwidth
- Zero crossing rate
- MFCCs (13 coefficients)
- Chroma features
- Fundamental frequency (f0) using YIN

Returns:
    df (pandas.DataFrame): one row per frame with all features
    sr (int): sample rate of loaded audio
"""
from typing import Tuple
import numpy as np
import pandas as pd
import librosa
from src.python.app.constants.constants import Constants

def extract_features_to_df(
    audio_path: str,
    frame_seconds: float = 1.0,
    hop_ratio: float = Constants.DEFAULT_HOP_RATIO,
    sr_override: int | None = None
) -> Tuple[pd.DataFrame, int]:
    # Load audio (mono)
    y, sr = librosa.load(str(audio_path), sr=sr_override, mono=True)
    frame_length = max(1, int(round(frame_seconds * sr)))
    hop_length = max(1, int(round(frame_length * hop_ratio)))

    # Pad audio if shorter than one frame
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    # Pre-compute frame-wise features
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=13)

    # Chroma (try/catch to handle small windows)
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    except Exception:
        chroma = None

    # Fundamental frequency (may fail on short or noisy signals)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=sr / 2, frame_length=frame_length, hop_length=hop_length)
    except Exception:
        f0 = None

    # Construct DataFrame
    rows = []
    n_frames = len(rms)
    for i in range(n_frames):
        start_s = float(i * hop_length / sr)
        end_s = float((i * hop_length + frame_length) / sr)
        row = {
            "frame_index": i,
            "frame_start_s": start_s,
            "frame_end_s": end_s,
            "rms": float(rms[i]),
            "spectral_centroid": float(cent[i]) if i < len(cent) else None,
            "spectral_bandwidth": float(band[i]) if i < len(band) else None,
            "zcr": float(zcr[i]) if i < len(zcr) else None,
            "frame_seconds": frame_seconds,
            "hop_seconds": hop_length / sr,
        }
        # MFCCs
        for m in range(mfccs.shape[0]):
            row[f"mfcc_{m+1}"] = float(mfccs[m, i]) if i < mfccs.shape[1] else None
        # Chroma
        if chroma is not None:
            for c in range(chroma.shape[0]):
                row[f"chroma_{c+1}"] = float(chroma[c, i]) if i < chroma.shape[1] else None
        # f0
        if f0 is not None and i < len(f0):
            val = f0[i]
            row["f0_hz"] = float(val) if not np.isnan(val) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, sr

