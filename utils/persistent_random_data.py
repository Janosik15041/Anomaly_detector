"""
persistent_random_data.py - Generate infinite realistic synthetic stock data

Generates continuous synthetic data in real-time that follows realistic stock patterns.
When selected from the dropdown, it provides an infinite stream of data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random


class SyntheticStockGenerator:
    """
    Generates infinite realistic synthetic stock data based on historical patterns
    """

    def __init__(self, reference_file=None, start_price=100.0):
        """
        Initialize the generator

        Args:
            reference_file: Path to CSV file to learn patterns from (optional)
            start_price: Starting price if no reference file
        """
        self.reference_data = None
        self.price_volatility = 0.02  # Default 2% volatility
        self.volume_mean = 10000000
        self.volume_std = 3000000
        self.trend_strength = 0.0  # No trend by default
        self.current_price = start_price
        self.last_timestamp = datetime.now()
        self.generated_candles = []  # Store all generated candles

        if reference_file and os.path.exists(reference_file):
            self._learn_from_reference(reference_file)

    def _learn_from_reference(self, file_path):
        """
        Analyze reference data to learn realistic patterns

        Args:
            file_path: Path to reference CSV file
        """
        try:
            df = pd.read_csv(file_path)

            # Handle datetime column
            if 'Datetime' not in df.columns and 'Date' in df.columns:
                df = df.rename(columns={'Date': 'Datetime'})

            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_localize(None)

            # Calculate statistics from reference data
            if len(df) > 1:
                # Price volatility (percentage change)
                df['returns'] = df['Close'].pct_change()
                self.price_volatility = df['returns'].std()

                # Volume statistics
                self.volume_mean = df['Volume'].mean()
                self.volume_std = df['Volume'].std()

                # Trend (overall direction)
                self.trend_strength = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] / len(df)

                # Starting price
                self.current_price = df['Close'].iloc[-1]

                # Last timestamp
                self.last_timestamp = df['Datetime'].iloc[-1]

                self.reference_data = df

                print(f"📊 Learned patterns from {file_path}")
                print(f"   - Price volatility: {self.price_volatility*100:.2f}%")
                print(f"   - Average volume: {self.volume_mean:,.0f}")
                print(f"   - Trend: {self.trend_strength*100:.4f}% per period")
                print(f"   - Starting price: ${self.current_price:.2f}")

        except Exception as e:
            print(f"⚠️  Could not learn from reference file: {e}")
            print(f"   Using default parameters")

    def generate_next_candle(self, interval_minutes=60):
        """
        Generate the next OHLCV candle and store it

        Args:
            interval_minutes: Time interval in minutes

        Returns:
            dict with keys: Datetime, Open, High, Low, Close, Volume
        """
        # Update timestamp
        self.last_timestamp += timedelta(minutes=interval_minutes)

        # Generate price movement with trend
        price_change_pct = np.random.normal(self.trend_strength, self.price_volatility)
        open_price = self.current_price
        close_price = open_price * (1 + price_change_pct)

        # Generate high and low with realistic spread
        volatility_range = abs(close_price - open_price) * random.uniform(1.2, 2.5)
        high_price = max(open_price, close_price) + random.uniform(0, volatility_range)
        low_price = min(open_price, close_price) - random.uniform(0, volatility_range)

        # Ensure high >= open/close and low <= open/close
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate volume (log-normal distribution for realism)
        volume = int(max(0, np.random.lognormal(
            np.log(self.volume_mean),
            self.volume_std / self.volume_mean
        )))

        # Update current price for next candle
        self.current_price = close_price

        candle = {
            'Datetime': self.last_timestamp,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        }

        # Store the generated candle
        self.generated_candles.append(candle)

        return candle

    def get_dataframe(self):
        """
        Get all generated candles as a DataFrame

        Returns:
            pandas DataFrame with all generated OHLCV data
        """
        if not self.generated_candles:
            # Generate initial candle if none exist
            self.generate_next_candle()

        return pd.DataFrame(self.generated_candles)

    def generate_batch(self, num_candles=100, interval_minutes=60):
        """
        Generate a batch of candles

        Args:
            num_candles: Number of candles to generate
            interval_minutes: Time interval between candles

        Returns:
            pandas DataFrame with OHLCV data
        """
        candles = []
        for _ in range(num_candles):
            candles.append(self.generate_next_candle(interval_minutes))

        return pd.DataFrame(candles)

    def reset(self):
        """Reset generator to initial state"""
        if self.reference_data is not None:
            self.current_price = self.reference_data['Close'].iloc[-1]
            self.last_timestamp = self.reference_data['Datetime'].iloc[-1]
        else:
            self.current_price = 100.0
            self.last_timestamp = datetime.now()


def create_synthetic_dataset(output_file, reference_file=None, num_candles=1000, interval_minutes=60):
    """
    Create a synthetic dataset and save to CSV

    Args:
        output_file: Path to save the generated CSV
        reference_file: Path to reference CSV for learning patterns
        num_candles: Number of candles to generate
        interval_minutes: Time interval between candles
    """
    generator = SyntheticStockGenerator(reference_file)
    data = generator.generate_batch(num_candles, interval_minutes)

    # Save to CSV
    data.to_csv(output_file, index=False)

    print(f"\n✅ Generated {num_candles} synthetic candles")
    print(f"   Saved to: {output_file}")
    print(f"   Date range: {data['Datetime'].iloc[0]} to {data['Datetime'].iloc[-1]}")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")


if __name__ == '__main__':
    # Example: Generate synthetic data based on AAPL patterns
    reference_files = [
        'data/AAPL_1y_hourly.csv',
        'data/MSFT_1y.csv',
        'data/GOOGL_1y.csv',
        'data/TSLA_1y.csv'
    ]

    # Find first available reference file
    reference_file = None
    for ref in reference_files:
        if os.path.exists(ref):
            reference_file = ref
            break

    if reference_file:
        create_synthetic_dataset(
            output_file='data/SYNTHETIC_continuous.csv',
            reference_file=reference_file,
            num_candles=2000,  # Generate 2000 hourly candles (~83 days)
            interval_minutes=60
        )
    else:
        print("⚠️  No reference files found. Creating with default parameters...")
        create_synthetic_dataset(
            output_file='data/SYNTHETIC_continuous.csv',
            num_candles=2000,
            interval_minutes=60
        )
