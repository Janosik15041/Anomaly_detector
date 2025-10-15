"""
Live data source module for fetching real-time stock data from yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime


class LiveDataSource:
    """Fetches live stock data from Yahoo Finance"""
    
    def __init__(self):
        self.last_fetch_time = None
        
    def fetch_data(self, ticker="AAPL", period="7d", interval="1m"):
        """
        Fetch recent stock data from yfinance
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'TSLA')
            period: Time period ('1d', '5d', '7d', '1mo')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            
        Returns:
            pandas DataFrame with OHLCV data, or None if fetch fails
        """
        try:
            print(f"Fetching live data for {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, interval=interval)
            
            if data.empty:
                print(f"Warning: No data returned for {ticker}")
                return None
            
            self.last_fetch_time = datetime.now()
            print(f"✓ Fetched {len(data)} data points for {ticker}")
            return data
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return None
    
    def get_latest_price(self, ticker="AAPL"):
        """
        Get the most recent stock price
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Latest close price, or None if fetch fails
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1d", interval="1m")
            
            if data.empty:
                return None
                
            return data['Close'].iloc[-1]
            
        except Exception as e:
            print(f"Error fetching latest price: {e}")
            return None
    
    def validate_ticker(self, ticker):
        """
        Check if a ticker symbol is valid
        
        Args:
            ticker: Stock symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False