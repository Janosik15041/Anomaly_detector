"""
main.py - Main Application Entry Point
Run: streamlit run main.py

Orchestrates the backend (streamer) and frontend (app.py)
"""
import streamlit as st
import pandas as pd
import os

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
    render_metrics,
    render_statistics,
    render_sidebar
)

# Main application logic
st.title("📈 Stock Anomaly Detector")
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

# Render sidebar and get settings
selected_file, chart_type = render_sidebar(data_files)

# Load selected data
file_path = os.path.join(data_dir, selected_file)
df = load_stock_data(file_path)

# Render metrics
render_metrics(df)

st.markdown("---")

# Render charts based on selection
if chart_type in ["Candlestick", "Both"]:
    st.subheader("📊 Candlestick Chart")
    fig = create_candlestick_chart(df)
    st.plotly_chart(fig, use_container_width=True)

if chart_type in ["Line Chart", "Both"]:
    st.subheader("📈 Line Chart")
    fig = create_line_chart(df)
    st.plotly_chart(fig, use_container_width=True)

# Render statistics
render_statistics(df)

# Data table
st.markdown("---")
with st.expander("📋 View Raw Data"):
    st.dataframe(df, use_container_width=True)

# Date range info in sidebar
st.sidebar.markdown("---")
st.sidebar.write("**Data Range**")
st.sidebar.write(f"From: {df['Datetime'].iloc[0]}")
st.sidebar.write(f"To: {df['Datetime'].iloc[-1]}")
