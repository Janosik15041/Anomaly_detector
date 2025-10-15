"""
main.py - Main Application Entry Point
Run: streamlit run main.py

Orchestrates the backend (streamer) and frontend (app.py)
"""
import streamlit as st
import pandas as pd
import os
import time

# Configure page (must be first Streamlit command)
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide"
)

# Import frontend components from app.py
from app import (
    load_stock_data,
    create_candlestick_chart,
    create_line_chart,
    render_statistics
)

# Main application logic
st.title("📈 Stock Anomaly Detector - Live Streaming")
st.markdown("---")

# Get available data files
data_dir = "data"
if not os.path.exists(data_dir):
    st.error("Data directory not found!")
    st.stop()

data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

if not data_files:
    st.error("No data files found in the data directory!")
    st.stop()

# Initialize session state
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'speed_multiplier' not in st.session_state:
    st.session_state.speed_multiplier = 1
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = data_files[0]
if 'window_size' not in st.session_state:
    st.session_state.window_size = 100  # Show last 100 data points

# Sidebar controls
st.sidebar.header("⚙️ Settings")

# File selection
selected_file = st.sidebar.selectbox(
    "Select Stock Data",
    data_files,
    index=data_files.index(st.session_state.selected_file),
    format_func=lambda x: x.replace('.csv', '').replace('_', ' ').upper()
)

# Update selected file if changed
if selected_file != st.session_state.selected_file:
    st.session_state.selected_file = selected_file
    st.session_state.current_index = 0
    st.session_state.streaming = False
    st.session_state.paused = False

# Chart type selection
chart_type = st.sidebar.radio(
    "Chart Type",
    ["Candlestick", "Line Chart", "Both"]
)

# Control buttons
st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Controls")

# Show current state
if st.session_state.streaming and not st.session_state.paused:
    st.sidebar.success("🔴 Streaming")
elif st.session_state.paused:
    st.sidebar.warning("⏸️ Paused")
else:
    st.sidebar.info("⏹️ Stopped")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_disabled = st.session_state.streaming and not st.session_state.paused
    if st.sidebar.button(
        "▶️ Start",
        use_container_width=True,
        type="primary" if not start_disabled else "secondary",
        disabled=start_disabled
    ):
        st.session_state.streaming = True
        st.session_state.paused = False
        st.rerun()

with col2:
    pause_disabled = not st.session_state.streaming
    if st.sidebar.button(
        "⏸️ Pause",
        use_container_width=True,
        type="primary" if st.session_state.paused else "secondary",
        disabled=pause_disabled
    ):
        st.session_state.paused = not st.session_state.paused
        st.rerun()

col3, col4 = st.sidebar.columns(2)
with col3:
    if st.sidebar.button("⏹️ Stop", use_container_width=True):
        st.session_state.streaming = False
        st.session_state.paused = False
        st.session_state.current_index = 0  # Reset on stop
        st.rerun()

with col4:
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        st.session_state.current_index = 0
        st.session_state.streaming = False
        st.session_state.paused = False
        st.rerun()

# Speed controls
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Playback Speed")

speed_options = {
    "1x": 1,
    "2x": 2,
    "3x": 3,
    "5x": 5,
    "10x": 10,
    "100x": 100
}

selected_speed = st.sidebar.radio(
    "Speed",
    list(speed_options.keys()),
    index=0,
    horizontal=True
)
st.session_state.speed_multiplier = speed_options[selected_speed]

st.sidebar.caption(f"⏱️ 1 data point every {1.0/st.session_state.speed_multiplier:.3f} seconds")

# Window size control
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chart Window")
window_size = st.sidebar.slider(
    "Data points visible",
    min_value=50,
    max_value=500,
    value=st.session_state.window_size,
    step=50
)
st.session_state.window_size = window_size
st.sidebar.caption(f"📈 Showing last {window_size} data points")

# Load full data
file_path = os.path.join(data_dir, selected_file)
full_data = load_stock_data(file_path)

# Display streaming info
st.sidebar.markdown("---")
st.sidebar.write("**Streaming Info**")
st.sidebar.write(f"Total Points: {len(full_data):,}")
st.sidebar.write(f"Current: {st.session_state.current_index:,}")
progress_pct = (st.session_state.current_index / len(full_data)) * 100 if len(full_data) > 0 else 0
st.sidebar.write(f"Progress: {progress_pct:.1f}%")

# Date range info
st.sidebar.markdown("---")
st.sidebar.write("**Data Range**")
st.sidebar.write(f"From: {full_data['Datetime'].iloc[0]}")
st.sidebar.write(f"To: {full_data['Datetime'].iloc[-1]}")

# Clear cache button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Main content area
metrics_placeholder = st.empty()
st.markdown("---")

# Create persistent chart containers based on chart type
if chart_type in ["Candlestick", "Both"]:
    st.subheader("📊 Candlestick Chart")
    candlestick_placeholder = st.empty()

if chart_type in ["Line Chart", "Both"]:
    st.subheader("📈 Line Chart")
    line_placeholder = st.empty()

stats_placeholder = st.empty()

# Show current streaming data
if st.session_state.current_index > 0:
    # Get data up to current index
    all_loaded_data = full_data.iloc[:st.session_state.current_index].copy()

    # Apply sliding window - show only last N points
    start_idx = max(0, len(all_loaded_data) - st.session_state.window_size)
    window_data = all_loaded_data.iloc[start_idx:].copy()

    # Show metrics (for all loaded data, not just window)
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Data Points Loaded", f"{len(all_loaded_data):,}")

        with col2:
            st.metric("Current Price", f"${all_loaded_data['Close'].iloc[-1]:.2f}")

        with col3:
            change = all_loaded_data['Close'].iloc[-1] - all_loaded_data['Close'].iloc[0]
            change_pct = (change / all_loaded_data['Close'].iloc[0]) * 100
            st.metric("Change", f"${change:.2f}", f"{change_pct:+.2f}%")

        with col4:
            st.metric("Avg Volume", f"{all_loaded_data['Volume'].mean():,.0f}")

    # Show charts with windowed data (without unique keys to reduce flickering)
    if chart_type in ["Candlestick", "Both"]:
        fig = create_candlestick_chart(window_data)
        candlestick_placeholder.plotly_chart(fig, use_container_width=True)

    if chart_type in ["Line Chart", "Both"]:
        fig = create_line_chart(window_data)
        line_placeholder.plotly_chart(fig, use_container_width=True)

    # Show statistics for windowed data
    with stats_placeholder.container():
        render_statistics(window_data)

# Streaming logic
if st.session_state.streaming and not st.session_state.paused:
    if st.session_state.current_index < len(full_data):
        # Calculate delay based on speed multiplier
        delay = 1.0 / st.session_state.speed_multiplier

        # For smoother updates, batch multiple points at higher speeds
        if st.session_state.speed_multiplier >= 100:
            increment = 20  # Add 20 points at a time for 100x speed
        elif st.session_state.speed_multiplier >= 10:
            increment = 10  # Add 10 points at a time for 10x+ speed
        elif st.session_state.speed_multiplier >= 5:
            increment = 5  # Add 5 points at a time for 5x+ speed
        elif st.session_state.speed_multiplier >= 2:
            increment = 2  # Add 2 points at a time for 2x+ speed
        else:
            increment = 1  # Add 1 point at a time for 1x speed

        # Increment index (but don't exceed data length)
        st.session_state.current_index = min(
            st.session_state.current_index + increment,
            len(full_data)
        )

        # Wait before next update
        time.sleep(delay)

        # Rerun to update display
        st.rerun()
    else:
        # Reached end of data
        st.session_state.streaming = False
        st.balloons()
