import streamlit as st
from src.python.app.constants.constants import Constants
import os
import pandas as pd
import plotly.graph_objects as go
import io
import json
from collections import defaultdict
from src.python.app.utils.data_utils import get_region_columns
from src.python.app.utils.extract_josn_from_text import *

def render_regional_blendshapes(df, structure, region, key_prefix):
    """Render blendshapes visualization for a specific region"""
    region_cols = get_region_columns(structure, region)
    available_cols = region_cols['blendshapes']
    
    if available_cols and 'frame' in df.columns:
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features", len(available_cols))
        with col2:
            mean_activation = df[available_cols].mean().mean()
            st.metric("Avg Activation", f"{mean_activation:.3f}")
        with col3:
            max_activation = df[available_cols].max().max()
            st.metric("Max Activation", f"{max_activation:.3f}")
        
        # Time series plot
        fig = go.Figure()
        for col in available_cols[:15]:  # Limit to top 15
            fig.add_trace(go.Scatter(
                x=df['frame'],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"{region.capitalize()} Region Blendshapes Over Time",
            xaxis_title="Frame",
            yaxis_title="Activation",
            hovermode='x unified',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_blendshape_{region}")
        
        # Top activated features
        st.markdown("**Top 5 Most Activated Features:**")
        means = df[available_cols].mean().sort_values(ascending=False).head(5)
        
        for idx, (feature, value) in enumerate(means.items()):
            st.write(f"{idx+1}. **{feature}**: {value:.3f}")
    else:
        st.info(f"No {region} blendshapes found in data")

def render_emotions_pain(df, structure, key_prefix):
    """Render Emotions and Pain visualization for all data (no region split)"""
    emotion_cols = structure['emotion_cols']
    if emotion_cols and 'frame' in df.columns:
        # Time series plot for all emotions
        fig = go.Figure()
        for emotion_col in emotion_cols:
            fig.add_trace(go.Scatter(
                x=df['frame'],
                y=df[emotion_col],
                mode='lines',
                name=emotion_col,
                line=dict(width=Constants.TWO)
            ))

        fig.update_layout(
            title="Emotions Over Time",
            xaxis_title="Frame",
            yaxis_title="Emotion Score",
            hovermode='x unified',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_emotion_timeseries")

        # Top activated emotions
        st.markdown("**Top 5 Most Activated Emotions:**")
        means = df[emotion_cols].mean().sort_values(ascending=False).head(5)
        for idx, (emotion, value) in enumerate(means.items()):
            st.write(f"{idx+1}. **{emotion}**: {value:.3f}")
    else:
        st.info("No emotion data found in data.")

    # Pain visualization (if present)
    pain_cols = structure.get('pain_cols', [])
    if pain_cols and 'frame' in df.columns:
        st.markdown("---")
        st.markdown("### 😣 Pain Over Time")
        fig = go.Figure()
        for pain_col in pain_cols:
            fig.add_trace(go.Scatter(
                x=df['frame'],
                y=df[pain_col],
                mode='lines',
                name=pain_col,
                line=dict(width=2)
            ))
        fig.update_layout(
            title="Pain Over Time",
            xaxis_title="Frame",
            yaxis_title="Pain Score",
            hovermode='x unified',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_pain_timeseries")
    elif pain_cols:
        st.info("Pain columns found but no 'frame' column.")
    else:
        st.info("No pain data found in data.")


def render_regional_aus(df, structure, region, key_prefix):
    """Render AUs visualization for a specific region"""
    region_cols = get_region_columns(structure, region)
    available_aus = region_cols['aus']
    
    if available_aus and 'frame' in df.columns:
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AUs Found", len(available_aus))
        with col2:
            mean_activation = df[available_aus].mean().mean()
            st.metric("Avg Activation", f"{mean_activation:.3f}")
        with col3:
            max_activation = df[available_aus].max().max()
            st.metric("Max Activation", f"{max_activation:.3f}")
        
        # Time series plot
        fig = go.Figure()
        for au_col in available_aus:
            fig.add_trace(go.Scatter(
                x=df['frame'],
                y=df[au_col],
                mode='lines',
                name=au_col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"{region.capitalize()} Region Action Units Over Time",
            xaxis_title="Frame",
            yaxis_title="AU Intensity",
            hovermode='x unified',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_au_timeseries_{region}")
        
        # Heatmap of AU activations
        st.markdown("**AU Activation Heatmap:**")
        
        # Sample data for better visualization
        sampled_aus = df[['frame'] + available_aus].iloc[::max(1, len(df)//50)]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=sampled_aus[available_aus].T.values,
            x=sampled_aus['frame'].values,
            y=available_aus,
            colorscale='Viridis',
            colorbar=dict(title="Intensity")
        ))
        
        fig_heatmap.update_layout(
            title=f"{region.capitalize()} AUs Activation Heatmap",
            xaxis_title="Frame",
            yaxis_title="Action Unit",
            height=max(300, len(available_aus) * 30)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True, key=f"{key_prefix}_au_heatmap_{region}")
        
        # Top activated AUs
        st.markdown("**Top 5 Most Activated AUs:**")
        means = df[available_aus].mean().sort_values(ascending=False).head(5)
        
        for idx, (au, value) in enumerate(means.items()):
            st.write(f"{idx+1}. **{au}**: {value:.3f}")
    else:
        st.info(f"No {region} Action Units found in data")




def render_regional_comparison(df, structure, frame_ranges, region):
    """Render regional analysis comparing different frame ranges"""
    region_cols = get_region_columns(structure, region)
    
    if not region_cols['blendshapes'] and not region_cols['aus']:
        st.info(f"No data available for {region} region")
        return
    
    st.markdown(f"### {region.capitalize()} Region Analysis")
    
    # Create tabs for blendshapes and AUs
    available_tabs = []
    if region_cols['blendshapes']:
        available_tabs.append("🎭 Blendshapes")
    if region_cols['aus']:
        available_tabs.append("🔢 Action Units")
    
    if not available_tabs:
        return
    
    region_tabs = st.tabs(available_tabs)
    tab_idx = 0
    
    # Blendshapes for this region
    if region_cols['blendshapes']:
        with region_tabs[tab_idx]:
            cols = region_cols['blendshapes']
            
            st.markdown(f"#### 🎭 {region.capitalize()} Blendshapes Across Frame Ranges")
            
            # Statistics comparison table
            st.markdown("**Statistics by Frame Range:**")
            
            stats_data = []
            for idx, (start, end) in enumerate(frame_ranges):
                range_df = df[(df['frame'] >= start) & (df['frame'] <= end)]
                if len(range_df) > 0 and cols:
                    stats_data.append({
                        'Range': f'{start}-{end}',
                        'Frames': len(range_df),
                        'Avg': f"{range_df[cols].mean().mean():.3f}",
                        'Max': f"{range_df[cols].max().max():.3f}",
                        'Min': f"{range_df[cols].min().min():.3f}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            # Combined time series for all ranges
            st.markdown("**All Frame Ranges Combined:**")
            
            # Filter df to only include frames in ranges
            mask = pd.Series(False, index=df.index)
            for start, end in frame_ranges:
                mask |= (df['frame'] >= start) & (df['frame'] <= end)
            filtered_df = df[mask].copy()
            
            # Get top 10 most variable features across all ranges
            if len(filtered_df) > 0 and cols:
                variability = filtered_df[cols].std().sort_values(ascending=False).head(10)
                top_features = variability.index.tolist()
                
                fig = go.Figure()
                
                # Color palette for ranges
                range_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
                
                for feature in top_features:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['frame'],
                        y=filtered_df[feature],
                        mode='lines',
                        name=feature,
                        line=dict(width=2)
                    ))
                
                # Add vertical lines to separate ranges
                for idx, (start, end) in enumerate(frame_ranges):
                    if idx > 0:
                        fig.add_vline(x=start, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    # Add range label
                    mid_point = (start + end) / 2
                    fig.add_annotation(
                        x=mid_point,
                        y=1,
                        yref='paper',
                        text=f"Range {idx+1}<br>{start}-{end}",
                        showarrow=False,
                        font=dict(size=10, color=range_colors[idx % len(range_colors)]),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                
                fig.update_layout(
                    title=f"{region.capitalize()} - Top 10 Variable Blendshapes",
                    xaxis_title="Frame",
                    yaxis_title="Activation",
                    height=450,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"regional_blend_{region}")
                
                # Comparison across ranges
                st.markdown("**Average Activation by Range:**")
                
                comparison_data = []
                for idx, (start, end) in enumerate(frame_ranges):
                    range_df = df[(df['frame'] >= start) & (df['frame'] <= end)]
                    if len(range_df) > 0:
                        for col in top_features[:5]:  # Top 5 for clarity
                            comparison_data.append({
                                'Range': f'Range {idx+1} ({start}-{end})',
                                'Feature': col,
                                'Avg_Activation': range_df[col].mean()
                            })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    
                    fig_comp = go.Figure()
                    
                    for feature in top_features[:5]:
                        feature_data = comp_df[comp_df['Feature'] == feature]
                        fig_comp.add_trace(go.Bar(
                            x=feature_data['Range'],
                            y=feature_data['Avg_Activation'],
                            name=feature,
                            text=[f"{v:.3f}" for v in feature_data['Avg_Activation']],
                            textposition='auto'
                        ))
                    
                    fig_comp.update_layout(
                        title=f"{region.capitalize()} - Comparison Across Ranges",
                        xaxis_title="Frame Range",
                        yaxis_title="Average Activation",
                        height=400,
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True, key=f"regional_blend_comp_{region}")
        
        tab_idx += 1
    
    # AUs for this region
    if region_cols['aus']:
        with region_tabs[tab_idx]:
            cols = region_cols['aus']
            
            st.markdown(f"#### 🔢 {region.capitalize()} Action Units Across Frame Ranges")
            
            # Statistics comparison table
            st.markdown("**Statistics by Frame Range:**")
            
            stats_data = []
            for idx, (start, end) in enumerate(frame_ranges):
                range_df = df[(df['frame'] >= start) & (df['frame'] <= end)]
                if len(range_df) > 0 and cols:
                    stats_data.append({
                        'Range': f'{start}-{end}',
                        'Frames': len(range_df),
                        'Avg': f"{range_df[cols].mean().mean():.3f}",
                        'Max': f"{range_df[cols].max().max():.3f}",
                        'Min': f"{range_df[cols].min().min():.3f}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            # Combined time series for all ranges
            st.markdown("**All Frame Ranges Combined:**")
            
            # Filter df to only include frames in ranges
            mask = pd.Series(False, index=df.index)
            for start, end in frame_ranges:
                mask |= (df['frame'] >= start) & (df['frame'] <= end)
            filtered_df = df[mask].copy()
            
            # Get top AUs by mean activation
            if len(filtered_df) > 0 and cols:
                top_aus = filtered_df[cols].mean().sort_values(ascending=False).head(10)
                top_au_names = top_aus.index.tolist()
                
                fig = go.Figure()
                
                range_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
                
                for au in top_au_names:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['frame'],
                        y=filtered_df[au],
                        mode='lines',
                        name=au,
                        line=dict(width=2)
                    ))
                
                # Add vertical lines and labels
                for idx, (start, end) in enumerate(frame_ranges):
                    if idx > 0:
                        fig.add_vline(x=start, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    mid_point = (start + end) / 2
                    fig.add_annotation(
                        x=mid_point,
                        y=1,
                        yref='paper',
                        text=f"Range {idx+1}<br>{start}-{end}",
                        showarrow=False,
                        font=dict(size=10, color=range_colors[idx % len(range_colors)]),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                
                fig.update_layout(
                    title=f"{region.capitalize()} - Top 10 Action Units",
                    xaxis_title="Frame",
                    yaxis_title="AU Intensity",
                    height=450,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"regional_au_{region}")
                
                # Heatmap for all ranges
                st.markdown("**AU Activation Heatmap:**")
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=filtered_df[top_au_names].T.values,
                    x=filtered_df['frame'].values,
                    y=top_au_names,
                    colorscale='Viridis',
                    colorbar=dict(title="Intensity")
                ))
                
                # Add range separators
                for idx, (start, end) in enumerate(frame_ranges[1:], 1):
                    fig_heat.add_vline(x=start, line_dash="dash", line_color="white", line_width=2)
                
                fig_heat.update_layout(
                    title=f"{region.capitalize()} - AU Heatmap Across Ranges",
                    xaxis_title="Frame",
                    yaxis_title="Action Unit",
                    height=max(300, len(top_au_names) * 30)
                )
                
                st.plotly_chart(fig_heat, use_container_width=True, key=f"regional_au_heat_{region}")
                
                # Comparison across ranges
                st.markdown("**Average Intensity by Range:**")
                
                comparison_data = []
                for idx, (start, end) in enumerate(frame_ranges):
                    range_df = df[(df['frame'] >= start) & (df['frame'] <= end)]
                    if len(range_df) > 0:
                        for au in top_au_names[:5]:
                            comparison_data.append({
                                'Range': f'Range {idx+1} ({start}-{end})',
                                'AU': au,
                                'Avg_Intensity': range_df[au].mean()
                            })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    
                    fig_comp = go.Figure()
                    
                    for au in top_au_names[:5]:
                        au_data = comp_df[comp_df['AU'] == au]
                        fig_comp.add_trace(go.Bar(
                            x=au_data['Range'],
                            y=au_data['Avg_Intensity'],
                            name=au,
                            text=[f"{v:.3f}" for v in au_data['Avg_Intensity']],
                            textposition='auto'
                        ))
                    
                    fig_comp.update_layout(
                        title=f"{region.capitalize()} - AU Comparison Across Ranges",
                        xaxis_title="Frame Range",
                        yaxis_title="Average Intensity",
                        height=400,
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True, key=f"regional_au_comp_{region}")


def render_prefilter_visualization(df, structure, frame_ranges, data_type="blendshapes"):
    """Render visualization for prefiltered frames"""
    if df is None or len(df) == 0:
        st.warning(f"No {data_type} data available for prefiltered frames")
        return
    
    # Show frame ranges
    st.info(f"📍 **Showing data for frames:** {frame_ranges}")
    
    # Statistics for prefiltered data
    col1, col2, col3, col4 = st.columns(4)
    
    cols_to_analyze = structure['blendshape_cols'] if data_type == "blendshapes" else structure['au_cols']
    
    with col1:
        st.metric("Filtered Frames", len(df))
    with col2:
        st.metric(f"{data_type.capitalize()}", len(cols_to_analyze))
    with col3:
        if cols_to_analyze:
            avg_val = df[cols_to_analyze].mean().mean()
            st.metric("Avg Value", f"{avg_val:.3f}")
    with col4:
        if cols_to_analyze:
            max_val = df[cols_to_analyze].max().max()
            st.metric("Max Value", f"{max_val:.3f}")
    
    # Show raw filtered data
    with st.expander("📊 View Filtered Data"):
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"📥 Download Filtered {data_type.capitalize()}",
            data=csv_buffer.getvalue(),
            file_name=f"prefiltered_{data_type}.csv",
            mime="text/csv",
            key=f"download_prefilter_{data_type}"
        )
    
    # Visualization by frame ranges
    st.markdown("---")
    st.markdown(f"### 📈 {data_type.capitalize()} Analysis by Frame Range")
    
    for idx, (start, end) in enumerate(frame_ranges):
        with st.expander(f"🔍 Frame Range {idx+1}: {start} - {end} ({end-start+1} frames)"):
            range_df = df[(df['frame'] >= start) & (df['frame'] <= end)]
            
            if len(range_df) == 0:
                st.warning(f"No data for this range")
                continue
            
            # Top activated features in this range
            if cols_to_analyze:
                st.markdown("**Most Activated Features in This Range:**")
                means = range_df[cols_to_analyze].mean().sort_values(ascending=False).head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=means.values,
                        y=means.index,
                        orientation='h',
                        marker=dict(
                            color=means.values,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Activation")
                        )
                    )
                ])
                
                fig.update_layout(
                    title=f"Top 10 Features (Frames {start}-{end})",
                    xaxis_title="Mean Activation",
                    yaxis_title="Feature",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key=f"prefilter_{data_type}_range_{idx}")
                
                # Time series for this range
                st.markdown("**Frame-by-Frame Analysis:**")
                
                # Select top 5 features for clarity
                top_features = means.head(5).index.tolist()
                
                fig_ts = go.Figure()
                for feature in top_features:
                    fig_ts.add_trace(go.Scatter(
                        x=range_df['frame'],
                        y=range_df[feature],
                        mode='lines+markers',
                        name=feature,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                
                fig_ts.update_layout(
                    title=f"Top 5 Features Over Time (Frames {start}-{end})",
                    xaxis_title="Frame",
                    yaxis_title="Activation",
                    hovermode='x unified',
                    height=350,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
                )
                st.plotly_chart(fig_ts, use_container_width=True, key=f"prefilter_{data_type}_ts_{idx}")
    
    # Regional analysis for all prefiltered frames
    if Constants.FRAME_KEY in df.columns and cols_to_analyze:
        st.divider()
        st.markdown(f"### 🎯 Regional Analysis (All Prefiltered Frames)")
        
        region_tabs = st.tabs(["👁️ Eyes", "👄 Mouth", "👃 Nose"])
        
        for reg_idx, (region, region_tab) in enumerate(zip(["eyes", "mouth", "nose"], region_tabs)):
            with region_tab:
                if data_type == "blendshapes":
                    render_regional_blendshapes(df, structure, region, f"prefilter_{data_type}")
                elif data_type == "aus":
                    render_regional_aus(df, structure, region, f"prefilter_{data_type}")



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




def render_log_card(log_entry):
    """Render individual log card with color coding."""
    agent = log_entry.get('agent', 'Unknown')
    timestamp = log_entry.get('timestamp', 'N/A')
    message = log_entry.get('message', '')
    
    colors = get_agent_color(agent)
    
    # Truncate long messages for preview
    preview_msg = message[:150] + "..." if len(message) > 150 else message
    
    html_card = f"""
    <div style="
        background-color: {colors['bg']};
        border-left: 4px solid {colors['border']};
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    " onmouseover="this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)'" 
       onmouseout="this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <div style="flex: 1;">
                <span style="
                    color: {colors['text']};
                    font-weight: 700;
                    font-size: 14px;
                    display: inline-block;
                    padding: 2px 8px;
                    background-color: {colors['border']};
                    color: white;
                    border-radius: 4px;
                    margin-right: 8px;
                ">{agent}</span>
                <span style="
                    color: #666;
                    font-size: 12px;
                    font-family: monospace;
                ">{timestamp}</span>
            </div>
        </div>
        <div style="
            color: #333;
            font-size: 13px;
            line-height: 1.5;
            background-color: rgba(255,255,255,0.5);
            padding: 8px;
            border-radius: 4px;
            word-break: break-word;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            max-height: 300px;
            overflow-y: auto;
        ">{preview_msg}</div>
    </div>
    """
    return html_card

def render_batch_section(batch_id, logs):
    """Render a batch section with all its logs."""
    st.markdown(f"### 📦 Batch {batch_id}")
    
    # Add stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Logs", len(logs))
    with col2:
        unique_agents = len(set(log['agent'] for log in logs))
        st.metric("Unique Agents", unique_agents)
    with col3:
        if logs:
            st.metric("Duration", f"{logs[0].get('timestamp', 'N/A')} → {logs[-1].get('timestamp', 'N/A')}")
    
    st.divider()
    
    # Filter options per batch
    agents_in_batch = sorted(list(set([log['agent'] for log in logs])))
    selected_agents = st.multiselect(
        "Filter by Agent (Batch):",
        agents_in_batch,
        default=agents_in_batch,
        key=f"batch_{batch_id}_filter"
    )
    
    # Display logs
    filtered_logs = [log for log in logs if log['agent'] in selected_agents]
    
    if filtered_logs:
        for i, log in enumerate(filtered_logs):
            html_card = render_log_card(log)
            st.markdown(html_card, unsafe_allow_html=True)
            
            # Add expander for full message
            with st.expander(f"📄 Full message - {log.get('agent', 'Unknown')}"):
                st.code(log.get('message', 'No message'), language="json")
    else:
        st.info("No logs match the selected filter")
    
    st.markdown("---")


def get_agent_color(agent_name):
    """Get color scheme for an agent, with fallback to default."""
    return Constants.AGENT_COLORS.get(agent_name, {
        "bg": "#F5F5F5",
        "border": "#CCCCCC",
        "text": "#333333"
    })


def render_audio_json_result(result_text, idx):
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
            for idx, item in enumerate(ev_list, Constants.ONE):
                title = f"Evidence {idx}: {item['speaker']} • {item['window_length_s']}s • Frame {item['frame_index']}"
                with st.expander(title):
                    features = item.get("features", {})
                    if isinstance(features, dict):
                        features_text = json.dumps(features, indent=Constants.TWO, ensure_ascii=False)
                    else:
                        features_text = to_text(features)
                    st.markdown(f"""
                    <div class="subcard">
                    <div style="font-weight:600; margin-bottom:6px;">Features</div>
                    <div class="code">{features_text}</div>
                    <div class="divider"></div>
                    <div style="font-weight:600; margin-bottom:6px;">Acoustic Claim</div>
                    <div>{item.get("acoustic_claim", "N/A")}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # 7) Download + Raw JSON
        pretty = json.dumps(data, indent=Constants.TWO, ensure_ascii=False)

        digest = hashlib.md5(pretty.encode("utf-8")).hexdigest()[:Constants.TEN]
        dl_key = f"{Constants.AUDIO_STR.lower()}{Constants.UNDERSCORE}{idx}{Constants.UNDERSCORE}{digest}"
        with st.expander("Raw JSON"):
            st.json(data)
        st.download_button("⬇️ Download JSON", pretty,
                        file_name="gemini_audio_analysis.json",
                        mime="application/json",
                        key = dl_key,
                        use_container_width=True)
