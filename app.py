"""
app.py - Frontend UI Components Only
All visual components, charts, and UI rendering functions.
This file contains NO business logic, only presentation.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


@st.cache_data
def load_stock_data(file_path):
    """Load stock data from CSV file - no gap filling for streaming"""
    df = pd.read_csv(file_path)

    # Handle different column name variations
    # Check if Datetime column exists, if not, use the first column or index
    if 'Datetime' not in df.columns:
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'Datetime'})
        elif df.columns[0] in ['datetime', 'timestamp', 'time']:
            df = df.rename(columns={df.columns[0]: 'Datetime'})

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')

    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Don't fill gaps - use raw data for streaming
    return df


def fill_time_gaps(df):
    """
    Fill missing time intervals in the data
    
    Args:
        df: DataFrame with Datetime column
        
    Returns:
        DataFrame with filled gaps
    """
    if len(df) < 2:
        return df
    
    # Detect the most common time interval
    time_diffs = df['Datetime'].diff().dropna()
    most_common_interval = time_diffs.mode()[0] if len(time_diffs) > 0 else pd.Timedelta(minutes=1)
    
    # Create a complete time range
    start_time = df['Datetime'].min()
    end_time = df['Datetime'].max()
    
    # Generate complete time index
    complete_index = pd.date_range(start=start_time, end=end_time, freq=most_common_interval)
    
    # Reindex the dataframe
    df = df.set_index('Datetime')
    df = df.reindex(complete_index)
    
    # Forward fill OHLC data (use last known values)
    df['Open'] = df['Open'].ffill()
    df['High'] = df['High'].ffill()
    df['Low'] = df['Low'].ffill()
    df['Close'] = df['Close'].ffill()

    # Fill volume with 0 for missing periods
    df['Volume'] = df['Volume'].fillna(0)

    # Reset index to get Datetime back as column
    df = df.reset_index()
    # The index column is automatically named based on the original index name
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'Datetime'})
    # If the datetime index already has a name, it keeps it
    
    return df


def detect_time_interval(df):
    """
    Detect the time interval of the data (minute, hour, day)
    
    Args:
        df: DataFrame with Datetime column
        
    Returns:
        str: 'minute', 'hour', or 'day'
    """
    if len(df) < 2:
        return 'day'
    
    # Calculate average time difference between consecutive rows
    time_diffs = df['Datetime'].diff().dropna()
    avg_diff = time_diffs.mean()
    
    # Determine interval
    if avg_diff < pd.Timedelta(minutes=5):
        return 'minute'
    elif avg_diff < pd.Timedelta(hours=2):
        return 'hour'
    else:
        return 'day'


def format_datetime_axis(df, interval_type):
    """
    Format datetime labels based on interval type
    
    Args:
        df: DataFrame with Datetime column
        interval_type: 'minute', 'hour', or 'day'
        
    Returns:
        tuple: (tickvals, ticktext, num_ticks)
    """
    total_points = len(df)
    
    # Determine number of ticks based on interval
    if interval_type == 'minute':
        num_ticks = min(15, total_points)  # Show ~15 labels for minutes
        date_format = '%m/%d %H:%M'
    elif interval_type == 'hour':
        num_ticks = min(20, total_points)  # Show ~20 labels for hours
        date_format = '%m/%d %H:%M'
    else:  # day
        num_ticks = min(10, total_points)  # Show ~10 labels for days
        date_format = '%Y-%m-%d'
    
    # Calculate tick spacing
    tick_spacing = max(1, total_points // num_ticks)
    tickvals = list(range(0, total_points, tick_spacing))
    
    # Ensure we include the last point
    if tickvals[-1] != total_points - 1:
        tickvals.append(total_points - 1)
    
    # Format tick labels
    ticktext = [df['Datetime'].iloc[i].strftime(date_format) for i in tickvals]
    
    return tickvals, ticktext, num_ticks


def render_sidebar(data_files):
    """
    Render sidebar controls

    Args:
        data_files: List of available CSV files

    Returns:
        tuple: (selected_file, chart_type)
    """
    st.sidebar.header("⚙️ Settings")

    selected_file = st.sidebar.selectbox(
        "Select Stock Data",
        data_files,
        format_func=lambda x: x.replace('.csv', '').replace('_', ' ').upper()
    )

    chart_type = st.sidebar.radio(
        "Chart Type",
        ["Candlestick", "Line Chart", "Both"]
    )

    return selected_file, chart_type


def render_metrics(df):
    """
    Render metric cards at the top of the page

    Args:
        df: DataFrame with stock data
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Data Points", f"{len(df):,}")

    with col2:
        st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")

    with col3:
        change = df['Close'].iloc[-1] - df['Close'].iloc[0]
        change_pct = (change / df['Close'].iloc[0]) * 100
        st.metric("Period Change", f"${change:.2f}", f"{change_pct:+.2f}%")

    with col4:
        st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")


def create_candlestick_chart(df):
    """
    Create candlestick chart with volume

    Args:
        df: DataFrame with stock data

    Returns:
        plotly Figure object
    """
    # Detect time interval
    interval_type = detect_time_interval(df)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'Volume')
    )

    # Create index-based x-axis to remove gaps
    x_data = list(range(len(df)))

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=x_data,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00CC96',
            decreasing_line_color='#EF553B'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['#EF553B' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#00CC96'
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=x_data,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    # Format datetime axis
    tickvals, ticktext, _ = format_datetime_axis(df, interval_type)

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )

    # Update axes
    fig.update_xaxes(
        title_text="Date",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-45,
        row=2, col=1
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_line_chart(df):
    """
    Create line chart for close prices

    Args:
        df: DataFrame with stock data

    Returns:
        plotly Figure object
    """
    # Detect time interval
    interval_type = detect_time_interval(df)
    
    fig = go.Figure()

    # Create index-based x-axis to remove gaps
    x_data = list(range(len(df)))

    fig.add_trace(go.Scatter(
        x=x_data,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#636EFA', width=2),
        fill='tozeroy',
        fillcolor='rgba(99, 110, 250, 0.1)'
    ))

    # Format datetime axis
    tickvals, ticktext, _ = format_datetime_axis(df, interval_type)

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_dark',
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45
        )
    )

    return fig


def render_statistics(df):
    """
    Render statistics section

    Args:
        df: DataFrame with stock data
    """
    st.markdown("---")
    st.subheader("📊 Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Price Statistics**")
        stats_df = pd.DataFrame({
            'Metric': ['Open', 'High', 'Low', 'Close'],
            'Min': [df['Open'].min(), df['High'].min(), df['Low'].min(), df['Close'].min()],
            'Max': [df['Open'].max(), df['High'].max(), df['Low'].max(), df['Close'].max()],
            'Mean': [df['Open'].mean(), df['High'].mean(), df['Low'].mean(), df['Close'].mean()],
            'Std Dev': [df['Open'].std(), df['High'].std(), df['Low'].std(), df['Close'].std()]
        })
        st.dataframe(stats_df.style.format({
            'Min': '${:.2f}',
            'Max': '${:.2f}',
            'Mean': '${:.2f}',
            'Std Dev': '${:.2f}'
        }), hide_index=True, use_container_width=True)

    with col2:
        st.write("**Volume Statistics**")
        vol_stats = pd.DataFrame({
            'Metric': ['Total Volume', 'Average Volume', 'Max Volume', 'Min Volume'],
            'Value': [
                f"{df['Volume'].sum():,.0f}",
                f"{df['Volume'].mean():,.0f}",
                f"{df['Volume'].max():,.0f}",
                f"{df['Volume'].min():,.0f}"
            ]
        })
        st.dataframe(vol_stats, hide_index=True, use_container_width=True)