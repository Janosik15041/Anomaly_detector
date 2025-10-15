#!/usr/bin/env python3
"""
Test script for live_data.py - fetch and display real-time stock data
"""
import sys
import os

# Add parent directory to path so we can import live_data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_data import LiveDataSource
import time


def test_live_data(ticker="AAPL", interval="1m", period="1d"):
    """Test fetching live data for a given ticker"""

    print(f"\n{'='*60}")
    print(f"Testing Live Data Source")
    print(f"Ticker: {ticker}")
    print(f"Interval: {interval} (1-minute data)")
    print(f"Period: {period}")
    print(f"{'='*60}\n")

    # Create data source
    data_source = LiveDataSource()

    # Validate ticker first
    print(f"Validating ticker '{ticker}'...")
    if not data_source.validate_ticker(ticker):
        print(f"❌ Invalid ticker: {ticker}")
        return

    print(f"✓ Ticker is valid\n")

    # Fetch data
    data = data_source.fetch_data(ticker=ticker, period=period, interval=interval)

    if data is None:
        print("❌ Failed to fetch data")
        return

    print(f"\n{'='*60}")
    print(f"Data Summary")
    print(f"{'='*60}")
    print(f"Total data points: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Last fetch time: {data_source.last_fetch_time}")

    # Display the most recent data points
    print(f"\n{'='*60}")
    print(f"Most Recent 10 Data Points")
    print(f"{'='*60}\n")
    print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))

    # Get latest price
    print(f"\n{'='*60}")
    print(f"Latest Price Information")
    print(f"{'='*60}")
    latest_price = data_source.get_latest_price(ticker)
    if latest_price:
        print(f"Latest Close Price: ${latest_price:.2f}")

    # Show statistics
    print(f"\n{'='*60}")
    print(f"Statistics")
    print(f"{'='*60}")
    print(f"Average Close: ${data['Close'].mean():.2f}")
    print(f"Min Close: ${data['Close'].min():.2f}")
    print(f"Max Close: ${data['Close'].max():.2f}")
    print(f"Total Volume: {data['Volume'].sum():,}")


def continuous_monitoring(ticker="AAPL", interval_seconds=60):
    """
    Continuously monitor a ticker by fetching new data periodically

    Args:
        ticker: Stock symbol to monitor
        interval_seconds: How often to fetch new data (in seconds)
    """
    print(f"\n{'='*60}")
    print(f"Continuous Monitoring Mode")
    print(f"Ticker: {ticker}")
    print(f"Fetching every {interval_seconds} seconds")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    data_source = LiveDataSource()

    try:
        while True:
            latest_price = data_source.get_latest_price(ticker)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            if latest_price:
                print(f"[{current_time}] {ticker}: ${latest_price:.2f}")
            else:
                print(f"[{current_time}] Failed to fetch price for {ticker}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "AAPL"  # Default ticker

    # Ask user what they want to do
    print("\nLive Data Tester")
    print("================")
    print("1. Fetch and display data (single fetch)")
    print("2. Continuous monitoring (fetches every 60 seconds)")

    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == "2":
        continuous_monitoring(ticker, interval_seconds=60)
    else:
        test_live_data(ticker, interval="1m", period="1d")
