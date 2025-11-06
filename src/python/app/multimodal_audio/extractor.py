"""
Extracts frame-level acoustic features using librosa and returns them as a pandas DataFrame.

Constants are imported from constants.py for centralized configuration.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import librosa
from src.python.app.constants.constants import Constants  # [Completed] unified import

def extract_features_to_df(
    audio_path: str,
    frame_seconds: float = Constants.DEFAULT_FRAME_SECONDS,
    hop_ratio: float = Constants.DEFAULT_HOP_RATIO,
    sr_override: int | None = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Extract frame-level acoustic features and return as DataFrame.

    Args:
        audio_path (str): Path to the audio file.
        frame_seconds (float): Frame size in seconds.
        hop_ratio (float): Hop length ratio relative to frame size.
        sr_override (int | None): Sampling rate override.

    Returns:
        Tuple[pd.DataFrame, int]: DataFrame of features, and sampling rate.
    """
    # Load audio (mono)
    y, sr = librosa.load(str(audio_path), sr=sr_override, mono=Constants.LIBROSA_MONO)
    frame_length = max(1, int(round(frame_seconds * sr)))
    hop_length = max(1, int(round(frame_length * hop_ratio)))

    # Pad audio if shorter than one frame
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    # Pre-compute frame-wise features
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]
    band = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=Constants.N_MFCC
    )

    # Chroma (try/catch to handle small windows)
    try:
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )
    except Exception:
        chroma = None

    # Fundamental frequency (may fail on short or noisy signals)
    try:
        f0 = librosa.yin(
            y,
            fmin=Constants.F0_MIN_HZ,
            fmax=sr / 2,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        f0 = None

    # Construct DataFrame
    rows = []
    n_frames = len(rms)
    for i in range(n_frames):
        start_s = float(i * hop_length / sr)
        end_s = float((i * hop_length + frame_length) / sr)
        row = {
            Constants.FEATURE_FRAME_INDEX: i,
            Constants.FEATURE_START_S: start_s,
            Constants.FEATURE_END_S: end_s,
            Constants.FEATURE_RMS: float(rms[i]),
            Constants.FEATURE_SPECTRAL_CENTROID: float(cent[i]) if i < len(cent) else None,
            Constants.FEATURE_SPECTRAL_BANDWIDTH: float(band[i]) if i < len(band) else None,
            Constants.FEATURE_ZCR: float(zcr[i]) if i < len(zcr) else None,
            Constants.FEATURE_FRAME_SECONDS_COL: frame_seconds,
            Constants.FEATURE_HOP_SECONDS: hop_length / sr,
        }

        # MFCCs
        for m in range(mfccs.shape[0]):
            row[f"{Constants.FEATURE_MFCC_PREFIX}{m+1}"] = (
                float(mfccs[m, i]) if i < mfccs.shape[1] else None
            )

        # Chroma
        if chroma is not None:
            for c in range(chroma.shape[0]):
                row[f"{Constants.FEATURE_CHROMA_PREFIX}{c+1}"] = (
                    float(chroma[c, i]) if i < chroma.shape[1] else None
                )

        # f0
        if f0 is not None and i < len(f0):
            val = f0[i]
            row[Constants.FEATURE_F0_HZ] = float(val) if not np.isnan(val) else None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, sr
