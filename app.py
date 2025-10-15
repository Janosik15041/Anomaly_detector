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
    """Load stock data from CSV file"""
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    return df


def render_sidebar(data_files):
    """
    Render sidebar controls

    Args:
        data_files: List of available CSV files

    Returns:
        tuple: (selected_file, chart_type)
    """
    st.sidebar.header("Settings")

    selected_file = st.sidebar.selectbox(
        "Select Stock Data",
        data_files,
        format_func=lambda x: x.replace('.csv', '').replace('_', ' ')
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
        st.metric("Total Data Points", len(df))

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
            name='Price'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green'
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

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    # Update x-axis to show datetime labels at intervals
    tick_spacing = max(1, len(df) // 10)  # Show ~10 labels
    tickvals = list(range(0, len(df), tick_spacing))
    ticktext = [df['Datetime'].iloc[i].strftime('%Y-%m-%d %H:%M') for i in tickvals]

    fig.update_xaxes(
        title_text="Date",
        tickvals=tickvals,
        ticktext=ticktext,
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
    fig = go.Figure()

    # Create index-based x-axis to remove gaps
    x_data = list(range(len(df)))

    fig.add_trace(go.Scatter(
        x=x_data,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))

    # Update x-axis to show datetime labels at intervals
    tick_spacing = max(1, len(df) // 10)  # Show ~10 labels
    tickvals = list(range(0, len(df), tick_spacing))
    ticktext = [df['Datetime'].iloc[i].strftime('%Y-%m-%d %H:%M') for i in tickvals]

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext
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
        }), hide_index=True)

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
        st.dataframe(vol_stats, hide_index=True)
