import streamlit as st
from src.python.app.constants.constants import Constants
import os
import pandas as pd
import plotly.graph_objects as go
import io
import json
from collections import defaultdict

def detect_csv_structure(df):
    """Detect what data is present in the CSV"""
    structure = {
        'has_frame': 'frame' in df.columns,
        'has_blendshapes': False,
        'has_aus': False,
        'has_emotions': False,
        'has_pain': False,
        'blendshape_cols': [],
        'au_cols': [],
        'emotion_cols': [],
        'pain_cols': []
    }
    
    # Common blendshape names
    blendshape_keywords = ['eye', 'mouth', 'brow', 'nose', 'cheek', 'jaw', 'Blink', 'Squint', 
                           'Smile', 'Frown', 'Pucker', 'Funnel', 'Roll', 'Shrug']
    
    # Detect blendshapes
    # for col in df.columns:
    #     if any(keyword in col for keyword in blendshape_keywords):
    #         structure['blendshape_cols'].append(col)
    structure['blendshape_cols'] = Constants.BLENDSHAPE_KEY[:-4]
    
    # Detect AUs
    structure['au_cols'] = [col for col in df.columns if col.startswith('AU') and 
                            any(char.isdigit() for char in col)]
    
    # Detect emotions
    emotion_keywords = ['Joy', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt', 
                        'Happy', 'Sad', 'Angry', 'Fearful']
    structure['emotion_cols'] = [col for col in df.columns if any(e in col for e in emotion_keywords)]
    
    # Detect pain
    pain_keywords = ['pain', 'Pain', 'PAIN', 'discomfort', 'Discomfort']
    structure['pain_cols'] = [col for col in df.columns if any(p in col for p in pain_keywords)]
    
    # Set boolean flags
    structure['has_blendshapes'] = len(structure['blendshape_cols']) > 0
    structure['has_aus'] = len(structure['au_cols']) > 0
    structure['has_emotions'] = len(structure['emotion_cols']) > 0
    structure['has_pain'] = len(structure['pain_cols']) > 0
    
    return structure


def get_region_columns(structure, region):
    """Get columns for a specific facial region"""
    region_map = {
        "eyes": ["eyeBlink", "eyeSquint", "browDown", "browInner", "browOuter", "eyeWide", 
                 "eyeLook", "eyeClose"],
        "mouth": ["mouthSmile", "mouthPucker", "jawOpen", "mouthFunnel", "mouthLower", 
                  "mouthPress", "mouthRoll", "mouthShrug", "mouthStretch", "mouthUpper", 
                  "mouthFrown", "mouthClose"],
        "nose": ["cheekSquint", "noseSneer", "cheekPuff"]
    }
    
    au_region_map = {
        "eyes": ["AU01", "AU02", "AU04", "AU05", "AU07", "AU41", "AU42", "AU43", "AU44", 
                 "AU45", "AU46", "AU61", "AU62", "AU63", "AU64"],
        "mouth": ["AU10", "AU12", "AU13", "AU14", "AU15", "AU16", "AU17", "AU18", "AU20", 
                  "AU22", "AU23", "AU24", "AU25", "AU26", "AU27", "AU28"],
        "nose": ["AU06", "AU09", "AU11"]
    }
    
    result = {
        'blendshapes': [],
        'aus': []
    }
    
    # Get blendshapes for region
    if region in region_map:
        keywords = region_map[region]
        result['blendshapes'] = [col for col in structure['blendshape_cols'] 
                                 if any(kw in col for kw in keywords)]
    
    # Get AUs for region
    if region in au_region_map:
        au_prefixes = au_region_map[region]
        result['aus'] = [col for col in structure['au_cols'] 
                         if any(col.startswith(au) for au in au_prefixes)]
    
    return result


def load_data_from_sources(session_state_data, original_df=None):
    """Load data from various sources and return structured data"""
    data = {
        'blendshapes_df': None,
        'aus_df': None,
        'emotions_df': None,
        'pain_df': None,
        'structure': None
    }
    
    # Try to load from original uploaded file
    if original_df is not None:
        df = original_df
    else:
        # Try from various session state paths
        paths_to_try = [
            session_state_data.get("filtered_blendshape_csv_path"),
            session_state_data.get("prefiltered_csv_path"),
            session_state_data.get("blendshape_csv_path")
        ]
        
        df = None
        for path in paths_to_try:
            if path and os.path.exists(path):
                df = pd.read_csv(path)
                break
    
    if df is None:
        return data
    
    # Detect structure
    structure = detect_csv_structure(df)
    data['structure'] = structure
    
    # Extract blendshapes
    if structure['has_blendshapes']:
        cols = Constants.BLENDSHAPE_HEADERS_KEY[:-4] #['frame'] + structure['blendshape_cols'] if structure['has_frame'] else structure['blendshape_cols']
        data['blendshapes_df'] = df[cols]
    
    # Extract AUs
    if structure['has_aus']:
        cols = ['frame'] + structure['au_cols'] if structure['has_frame'] else structure['au_cols']
        data['aus_df'] = df[cols]
    
    # Extract emotions
    if structure['has_emotions']:
        cols = ['frame'] + structure['emotion_cols'] if structure['has_frame'] else structure['emotion_cols']
        data['emotions_df'] = df[cols]
    
    # Extract pain
    if structure['has_pain']:
        cols = ['frame'] + structure['pain_cols'] if structure['has_frame'] else structure['pain_cols']
        data['pain_df'] = df[cols]
    
    return data




def extract_frames_from_message(message):
    """Extract frame ranges from orchestrator/agent messages"""
    import re
    # Pattern to match [[21, 25], [50, 57], [60, 83]] or similar formats
    pattern = r'\[\s*\[([^\]]+)\]\s*(?:,\s*\[([^\]]+)\])*\s*\]'
    match = re.search(pattern, message)
    
    if match:
        # Extract all number pairs
        pairs_pattern = r'\[(\d+),\s*(\d+)\]'
        pairs = re.findall(pairs_pattern, message)
        frame_ranges = [[int(start), int(end)] for start, end in pairs]
        return frame_ranges
    
    return None




def parse_prefilter_decision(session_state_data):
    """Parse prefilter decision from session state"""
    prefilter_decision = session_state_data.get("prefilter_decision", "")
    
    try:
        # Try to extract JSON from the decision
        import re
        match = re.search(r"```json\s*(\{.*?\})\s*```", prefilter_decision, re.DOTALL)
        if match:
            decision = json.loads(match.group(1))
        else:
            # Try direct JSON parse
            decision = json.loads(prefilter_decision)
        
        return {
            'useful': decision.get('useful', False),
            'frame_ranges': decision.get('frame_ranges', []),
            'reason': decision.get('reason', 'No reason provided')
        }
    except Exception as e:
        return None


def get_prefiltered_frames(df, frame_ranges):
    """Extract only the frames specified in frame_ranges"""
    if 'frame' not in df.columns:
        return df
    
    keep_indices = []
    for start, end in frame_ranges:
        keep_indices.extend(range(start, end + 1))
    
    return df[df['frame'].isin(keep_indices)]


def render_frame_range_detailed_analysis(df, structure, start, end, range_idx):
    """Render detailed analysis for a specific frame range"""
    range_df = df[(df['frame'] >= start) & (df['frame'] <= end)].copy()
    
    if len(range_df) == 0:
        st.warning(f"No data for frames {start}-{end}")
        return
    
    # Create sub-tabs for different data types
    available_tabs = []
    if structure['has_blendshapes']:
        available_tabs.append("🎭 Blendshapes")
    if structure['has_aus']:
        available_tabs.append("🔢 Action Units")
    if structure['has_emotions']:
        available_tabs.append("😊 Emotions")
    if structure['has_pain']:
        available_tabs.append("😣 Pain")
    
    if not available_tabs:
        st.warning("No data available for this range")
        return
    
    range_tabs = st.tabs(available_tabs)
    tab_idx = 0
    
    # Blendshapes Analysis
    if structure['has_blendshapes']:
        with range_tabs[tab_idx]:
            st.markdown(f"#### 🎭 Blendshapes (Frames {start}-{end})")
            
            cols = structure['blendshape_cols']
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Features", len(cols))
            with col2:
                avg_val = range_df[cols].mean().mean()
                st.metric("Avg Activation", f"{avg_val:.3f}")
            with col3:
                max_val = range_df[cols].max().max()
                st.metric("Max Activation", f"{max_val:.3f}")
            
            # Top 10 activated
            st.markdown("**Top 10 Activated Blendshapes:**")
            top_10 = range_df[cols].mean().sort_values(ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=top_10.index,
                    x=top_10.values,
                    orientation='h',
                    marker=dict(
                        color=top_10.values,
                        colorscale='Blues',
                        showscale=True,
                        colorbar=dict(title="Activation")
                    ),
                    text=[f"{v:.3f}" for v in top_10.values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Top 10 Blendshapes (Frames {start}-{end})",
                xaxis_title="Mean Activation",
                yaxis_title="Feature",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key=f"blend_bar_{range_idx}")
            
            # Time series for top 5
            st.markdown("**Top 5 Blendshapes Over Time:**")
            top_5 = top_10.head(5).index.tolist()
            
            fig_ts = go.Figure()
            for feature in top_5:
                fig_ts.add_trace(go.Scatter(
                    x=range_df['frame'],
                    y=range_df[feature],
                    mode='lines+markers',
                    name=feature,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig_ts.update_layout(
                title=f"Blendshapes Timeline (Frames {start}-{end})",
                xaxis_title="Frame",
                yaxis_title="Activation",
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig_ts, use_container_width=True, key=f"blend_ts_{range_idx}")
            
            # CSV data
            with st.expander("📄 View Blendshapes CSV"):
                display_df = range_df[['frame'] + cols]
                st.dataframe(display_df, use_container_width=True)
                
                csv_buffer = io.StringIO()
                display_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Download Blendshapes (Frames {start}-{end})",
                    data=csv_buffer.getvalue(),
                    file_name=f"blendshapes_frames_{start}_{end}.csv",
                    mime="text/csv",
                    key=f"dl_blend_{range_idx}"
                )
        tab_idx += 1
    
    # Action Units Analysis
    if structure['has_aus']:
        with range_tabs[tab_idx]:
            st.markdown(f"#### 🔢 Action Units (Frames {start}-{end})")
            
            cols = structure['au_cols']
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AUs", len(cols))
            with col2:
                avg_val = range_df[cols].mean().mean()
                st.metric("Avg Intensity", f"{avg_val:.3f}")
            with col3:
                max_val = range_df[cols].max().max()
                st.metric("Max Intensity", f"{max_val:.3f}")
            
            # Top 10 activated AUs
            st.markdown("**Top 10 Activated Action Units:**")
            top_10 = range_df[cols].mean().sort_values(ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=top_10.index,
                    x=top_10.values,
                    orientation='h',
                    marker=dict(
                        color=top_10.values,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Intensity")
                    ),
                    text=[f"{v:.3f}" for v in top_10.values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Top 10 AUs (Frames {start}-{end})",
                xaxis_title="Mean Intensity",
                yaxis_title="Action Unit",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key=f"au_bar_{range_idx}")
            
            # Time series for top 5 AUs
            st.markdown("**Top 5 AUs Over Time:**")
            top_5 = top_10.head(5).index.tolist()
            
            fig_ts = go.Figure()
            for au in top_5:
                fig_ts.add_trace(go.Scatter(
                    x=range_df['frame'],
                    y=range_df[au],
                    mode='lines+markers',
                    name=au,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig_ts.update_layout(
                title=f"AU Timeline (Frames {start}-{end})",
                xaxis_title="Frame",
                yaxis_title="Intensity",
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig_ts, use_container_width=True, key=f"au_ts_{range_idx}")
            
            # Heatmap
            st.markdown("**AU Activation Heatmap:**")
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=range_df[cols].T.values,
                x=range_df['frame'].values,
                y=cols,
                colorscale='Viridis',
                colorbar=dict(title="Intensity")
            ))
            
            fig_heat.update_layout(
                title=f"AU Heatmap (Frames {start}-{end})",
                xaxis_title="Frame",
                yaxis_title="Action Unit",
                height=max(300, len(cols) * 20)
            )
            st.plotly_chart(fig_heat, use_container_width=True, key=f"au_heat_{range_idx}")
            
            # CSV data
            with st.expander("📄 View AUs CSV"):
                display_df = range_df[['frame'] + cols]
                st.dataframe(display_df, use_container_width=True)
                
                csv_buffer = io.StringIO()
                display_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Download AUs (Frames {start}-{end})",
                    data=csv_buffer.getvalue(),
                    file_name=f"action_units_frames_{start}_{end}.csv",
                    mime="text/csv",
                    key=f"dl_au_{range_idx}"
                )
        tab_idx += 1
    
    # Emotions Analysis
    if structure['has_emotions']:
        with range_tabs[tab_idx]:
            st.markdown(f"#### 😊 Emotions (Frames {start}-{end})")
            
            cols = structure['emotion_cols']
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emotions", len(cols))
            with col2:
                avg_val = range_df[cols].mean().mean()
                st.metric("Avg Score", f"{avg_val:.3f}")
            with col3:
                max_val = range_df[cols].max().max()
                st.metric("Max Score", f"{max_val:.3f}")
            
            # Average emotion scores
            st.markdown("**Average Emotion Scores:**")
            avg_emotions = range_df[cols].mean().sort_values(ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=avg_emotions.index,
                    x=avg_emotions.values,
                    orientation='h',
                    marker=dict(
                        color=['#4CAF50', '#FFC107', '#FF5722', '#9C27B0', '#F44336', '#795548', '#607D8B'][:len(cols)],
                    ),
                    text=[f"{v:.3f}" for v in avg_emotions.values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Emotion Scores (Frames {start}-{end})",
                xaxis_title="Mean Score",
                yaxis_title="Emotion",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key=f"emo_bar_{range_idx}")
            
            # Time series for all emotions
            st.markdown("**Emotions Over Time:**")
            
            fig_ts = go.Figure()
            emotion_colors = {
                'Joy': '#4CAF50', 'Happiness': '#4CAF50', 'Happy': '#4CAF50',
                'Sadness': '#2196F3', 'Sad': '#2196F3',
                'Anger': '#F44336', 'Angry': '#F44336',
                'Fear': '#9C27B0', 'Fearful': '#9C27B0',
                'Surprise': '#FFC107',
                'Disgust': '#795548',
                'Contempt': '#607D8B'
            }
            
            for emotion in cols:
                color = emotion_colors.get(emotion, '#999')
                fig_ts.add_trace(go.Scatter(
                    x=range_df['frame'],
                    y=range_df[emotion],
                    mode='lines+markers',
                    name=emotion,
                    line=dict(width=3, color=color),
                    marker=dict(size=8)
                ))
            
            fig_ts.update_layout(
                title=f"Emotion Timeline (Frames {start}-{end})",
                xaxis_title="Frame",
                yaxis_title="Score",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_ts, use_container_width=True, key=f"emo_ts_{range_idx}")
            
            # Dominant emotion per frame
            st.markdown("**Dominant Emotion per Frame:**")
            range_df['dominant_emotion'] = range_df[cols].idxmax(axis=1)
            range_df['dominant_score'] = range_df[cols].max(axis=1)
            
            fig_dom = go.Figure(data=[
                go.Scatter(
                    x=range_df['frame'],
                    y=range_df['dominant_score'],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=range_df['dominant_emotion'].map(emotion_colors),
                        line=dict(width=2, color='white')
                    ),
                    text=range_df['dominant_emotion'],
                    textposition='top center',
                    textfont=dict(size=9)
                )
            ])
            
            fig_dom.update_layout(
                title=f"Dominant Emotion (Frames {start}-{end})",
                xaxis_title="Frame",
                yaxis_title="Dominant Score",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_dom, use_container_width=True, key=f"emo_dom_{range_idx}")
            
            # CSV data
            with st.expander("📄 View Emotions CSV"):
                display_df = range_df[['frame'] + cols]
                st.dataframe(display_df, use_container_width=True)
                
                csv_buffer = io.StringIO()
                display_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Download Emotions (Frames {start}-{end})",
                    data=csv_buffer.getvalue(),
                    file_name=f"emotions_frames_{start}_{end}.csv",
                    mime="text/csv",
                    key=f"dl_emo_{range_idx}"
                )
        tab_idx += 1
    
    # Pain Analysis
    if structure['has_pain']:
        with range_tabs[tab_idx]:
            st.markdown(f"#### 😣 Pain (Frames {start}-{end})")
            
            cols = structure['pain_cols']
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pain Metrics", len(cols))
            with col2:
                avg_val = range_df[cols].mean().mean()
                st.metric("Avg Pain", f"{avg_val:.3f}")
            with col3:
                max_val = range_df[cols].max().max()
                st.metric("Max Pain", f"{max_val:.3f}")
            
            # Pain over time
            st.markdown("**Pain Intensity Over Time:**")
            
            fig_ts = go.Figure()
            for pain_col in cols:
                fig_ts.add_trace(go.Scatter(
                    x=range_df['frame'],
                    y=range_df[pain_col],
                    mode='lines+markers',
                    name=pain_col,
                    line=dict(width=3, color='#F44336'),
                    marker=dict(size=10),
                    fill='tozeroy',
                    fillcolor='rgba(244, 67, 54, 0.2)'
                ))
            
            fig_ts.update_layout(
                title=f"Pain Timeline (Frames {start}-{end})",
                xaxis_title="Frame",
                yaxis_title="Pain Intensity",
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig_ts, use_container_width=True, key=f"pain_ts_{range_idx}")
            
            # Pain distribution
            st.markdown("**Pain Intensity Distribution:**")
            
            for pain_col in cols:
                fig_hist = go.Figure(data=[
                    go.Histogram(
                        x=range_df[pain_col],
                        nbinsx=20,
                        marker=dict(color='#F44336', line=dict(color='white', width=1))
                    )
                ])
                
                fig_hist.update_layout(
                    title=f"{pain_col} Distribution",
                    xaxis_title="Pain Intensity",
                    yaxis_title="Frequency",
                    height=250,
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True, key=f"pain_hist_{range_idx}_{pain_col}")
            
            # CSV data
            with st.expander("📄 View Pain CSV"):
                display_df = range_df[['frame'] + cols]
                st.dataframe(display_df, use_container_width=True)
                
                csv_buffer = io.StringIO()
                display_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Download Pain (Frames {start}-{end})",
                    data=csv_buffer.getvalue(),
                    file_name=f"pain_frames_{start}_{end}.csv",
                    mime="text/csv",
                    key=f"dl_pain_{range_idx}"
                )

def group_logs_by_batch(batch_data):
    """Group logs by batch_id."""
    batches = defaultdict(list)
    for entry in batch_data:
        batch_id = entry.get('batch_id', 'unknown')
        batches[batch_id].extend(entry.get('agent_logs', []))
    return dict(sorted(batches.items()))
 
