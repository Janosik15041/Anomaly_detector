import yfinance as yf
import pandas as pd

# Configuration
TICKER = "AAPL"
PERIOD = "1y"  # 5 days
INTERVAL = "1h"  # 1 minute intervals

# Fetch data
ticker = yf.Ticker(TICKER)
data = ticker.history(period=PERIOD, interval=INTERVAL)

# Save with ticker_period format
filename = f'data/{TICKER}_{PERIOD}.csv'
data.to_csv(filename)

print(f" Data saved to {filename}")
print(f"  - Rows: {len(data)}")
print(f"  - Date range: {data.index[0]} to {data.index[-1]}")