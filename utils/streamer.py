"""
Data streaming orchestrator - handles both live and historic data sources
"""
import pandas as pd
import numpy as np
from utils.live_data import LiveDataSource
import os
import time


class DataStreamer:
    """Orchestrates data streaming from live or historic sources"""
    
    def __init__(self, mode='live', ticker='AAPL', stream_speed=0.1):
        """
        Initialize the data streamer
        
        Args:
            mode: 'live' for yfinance data, 'historic' for CSV data
            ticker: Stock symbol
            stream_speed: Delay between data points in seconds
        """
        self.mode = mode
        self.ticker = ticker
        self.stream_speed = stream_speed
        self.live_source = LiveDataSource()
        self.data = None
        self.current_index = 0
        
    def load_data(self):
        """Load data based on the selected mode"""
        if self.mode == 'live':
            return self._load_live_data()
        else:
            return self._load_historic_data()
    
    def _load_live_data(self):
        """Fetch live data from yfinance"""
        print(f"Loading live data for {self.ticker}...")
        data = self.live_source.fetch_data(self.ticker, period="7d", interval="1m")
        
        if data is None or data.empty:
            print("Failed to fetch live data, falling back to historic data...")
            return self._load_historic_data()
        
        self.data = data
        return True
    
    def _load_historic_data(self):
        """Load historic data from CSV files"""
        print(f"Loading historic data for {self.ticker}...")
        
        # Look for CSV files in data/ directory
        data_dir = 'data'
        if not os.path.exists(data_dir):
            print(f"Error: {data_dir} directory not found")
            return False
        
        # Try to find a matching CSV file
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        # Look for ticker-specific file first
        ticker_files = [f for f in csv_files if f.startswith(self.ticker)]
        
        if ticker_files:
            csv_file = os.path.join(data_dir, ticker_files[0])
        elif csv_files:
            # Use any available CSV file
            csv_file = os.path.join(data_dir, csv_files[0])
            print(f"No CSV for {self.ticker}, using {csv_files[0]}")
        else:
            print(f"Error: No CSV files found in {data_dir}")
            return False
        
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            self.data = df
            print(f"✓ Loaded {len(df)} data points from {csv_file}")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def get_all_data(self):
        """Get all loaded data at once (for initial display)"""
        if self.data is None:
            if not self.load_data():
                return None
        return self.data
    
    def stream_data(self):
        """
        Generator that yields data points one at a time
        
        Yields:
            dict: {
                'timestamp': datetime,
                'close': float,
                'open': float,
                'high': float,
                'low': float,
                'volume': int,
                'index': int
            }
        """
        if self.data is None:
            if not self.load_data():
                return
        
        for idx, (timestamp, row) in enumerate(self.data.iterrows()):
            yield {
                'timestamp': timestamp,
                'close': row['Close'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'volume': row['Volume'],
                'index': idx
            }
            time.sleep(self.stream_speed)
    
    def get_data_subset(self, start_idx, end_idx):
        """
        Get a subset of data by index
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            
        Returns:
            DataFrame subset
        """
        if self.data is None:
            if not self.load_data():
                return None
        
        return self.data.iloc[start_idx:end_idx]
    
    def get_close_prices(self):
        """Get just the close prices as a numpy array"""
        if self.data is None:
            if not self.load_data():
                return None
        
        return self.data['Close'].values
    
    def reset(self):
        """Reset the streamer to start from beginning"""
        self.current_index = 0
    
    def switch_mode(self, new_mode):
        """
        Switch between live and historic mode
        
        Args:
            new_mode: 'live' or 'historic'
        """
        if new_mode != self.mode:
            self.mode = new_mode
            self.data = None
            self.current_index = 0
            print(f"Switched to {new_mode} mode")
    
    def change_ticker(self, new_ticker):
        """
        Change the stock ticker
        
        Args:
            new_ticker: New stock symbol
        """
        if new_ticker != self.ticker:
            self.ticker = new_ticker
            self.data = None
            self.current_index = 0
            print(f"Changed ticker to {new_ticker}")