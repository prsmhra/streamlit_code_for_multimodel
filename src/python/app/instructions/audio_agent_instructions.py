SAMPLE_PROMPT = f"""
You are a careful, conservative clinical conversational analyst with expertise in speech acoustic markers.
DO NOT use any local heuristics — use ONLY the raw audio file and the provided per-window feature files.
We have also provided automated RandomForest feature-importance guidance per window (top features with importance scores).
The RF guidance is a local, reproducible, transparent suggestion about which features were most predictive of short-term RMS in each window; you MUST NOT treat RF guidance as definitive ground truth, but you MAY use it to prioritize frames/features to inspect.

I have provided structured per-frame/per-window acoustic feature files extracted at multiple window sizes.
Each feature frame contains (when available): rms, spectral_centroid, spectral_bandwidth, zcr, mfcc_1..13, chroma_1..12, f0_hz, frame_start_s, frame_end_s.

Task:
1) Use the raw audio alongside ALL provided feature JSON files (and the RF guidance) to analyze whether either speaker shows acoustic patterns that could be suggestive of mental-health struggles.
2) For EVERY claim you make you MUST cite one or more exact frames from the feature files (include window_length_s and frame_index) and the top features, show the numeric values for the key features from those frames, and explain in 1-2 sentences how those numeric values support the acoustic claim.
3) If the same timestamp range appears in multiple window-size files, explicitly compare the numeric values across the windows and state whether they agree or disagree and whether that raises or lowers confidence.
4) Combine complementary features when indicating voice quality/prosody (e.g., loudness: rms + spectral_centroid + spectral_bandwidth; pitch: f0_hz plus mfcc_1..3; variability: short-term changes across frames).
5) If you use the RF guidance, state the window and feature name and its RF importance score in your evidence. Do NOT invent numbers.
6) If you can infer speaker turns reliably, set 'speaker' to 'Speaker A' or 'Speaker B' in evidence items; otherwise use 'unknown'.

Output format: Return EXACTLY one valid JSON object (no surrounding text). It must contain these keys:
- "summary" : string (3-4 sentences)
- "per_speaker_findings" : object mapping speaker -> conclusions/evidence
- "evidence_list" : array of evidence items (include numeric feature values)
- "confidence_assessment" : string

Important constraints:
- Use the PROVIDED JSON files for numeric evidence. Do not invent numbers.
- Be conservative and non-diagnostic.
- Return ONLY valid JSON (no commentary, no markdown).
"""