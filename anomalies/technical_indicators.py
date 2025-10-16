"""
Technical Indicator Based Anomaly Rules

Advanced rules based on common technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volatility measures
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_rule import BaseAnomalyRule


class RSIExtremeRule(BaseAnomalyRule):
    """
    Detects extreme RSI conditions (overbought/oversold)

    RSI > 70: Overbought (potential reversal down)
    RSI < 30: Oversold (potential reversal up)
    """

    def __init__(self):
        super().__init__(
            name="RSI Extreme",
            description="Detects overbought (RSI>70) or oversold (RSI<30) conditions",
            severity=2,
            enabled=True,
            window_size=14,
            cooldown_periods=5,
            parameters={
                'rsi_period': 14,
                'overbought_threshold': 70,
                'oversold_threshold': 30,
                'extreme_overbought': 80,
                'extreme_oversold': 20
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate RSI extremes"""

        if not self.check_cooldown(current_index):
            return None

        rsi_period = self.parameters['rsi_period']
        if current_index < rsi_period + 1:
            return None

        # Get price data for RSI calculation
        price_data = data['Close'].iloc[:current_index + 1]
        rsi = self.calculate_rsi(price_data, period=rsi_period)

        current_row = data.iloc[current_index]

        # Check for overbought
        if rsi >= self.parameters['overbought_threshold']:
            condition = "overbought"
            severity = self.severity

            if rsi >= self.parameters['extreme_overbought']:
                condition = "extremely overbought"
                severity = min(4, severity + 1)

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': rsi,
                'baseline': 50,  # Neutral RSI
                'deviation': abs(rsi - 50),
                'details': (
                    f"RSI {condition}: {rsi:.1f} "
                    f"(threshold: {self.parameters['overbought_threshold']})"
                ),
                'metrics': {
                    'rsi': rsi,
                    'condition': condition,
                    'current_price': current_row['Close']
                }
            }

        # Check for oversold
        elif rsi <= self.parameters['oversold_threshold']:
            condition = "oversold"
            severity = self.severity

            if rsi <= self.parameters['extreme_oversold']:
                condition = "extremely oversold"
                severity = min(4, severity + 1)

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': rsi,
                'baseline': 50,
                'deviation': abs(50 - rsi),
                'details': (
                    f"RSI {condition}: {rsi:.1f} "
                    f"(threshold: {self.parameters['oversold_threshold']})"
                ),
                'metrics': {
                    'rsi': rsi,
                    'condition': condition,
                    'current_price': current_row['Close']
                }
            }

        return None


class MACDSignalRule(BaseAnomalyRule):
    """
    Detects MACD signal line crossovers

    MACD crossing above signal: Bullish
    MACD crossing below signal: Bearish
    """

    def __init__(self):
        super().__init__(
            name="MACD Signal",
            description="Detects MACD signal line crossovers (bullish/bearish signals)",
            severity=2,
            enabled=True,
            window_size=26,
            cooldown_periods=8,
            parameters={
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'min_histogram': 0.5  # Minimum histogram value to consider significant
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate MACD crossover"""

        if not self.check_cooldown(current_index):
            return None

        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']

        if current_index < slow_period + signal_period + 1:
            return None

        # Calculate current MACD
        prices = data['Close'].iloc[:current_index + 1]
        macd_curr, signal_curr, histogram_curr = self.calculate_macd(
            prices,
            fast=self.parameters['fast_period'],
            slow=slow_period,
            signal=signal_period
        )

        # Calculate previous MACD
        prices_prev = data['Close'].iloc[:current_index]
        macd_prev, signal_prev, histogram_prev = self.calculate_macd(
            prices_prev,
            fast=self.parameters['fast_period'],
            slow=slow_period,
            signal=signal_period
        )

        current_row = data.iloc[current_index]

        # Detect crossover
        crossed_above = histogram_prev <= 0 and histogram_curr > 0
        crossed_below = histogram_prev >= 0 and histogram_curr < 0

        if (crossed_above or crossed_below) and abs(histogram_curr) >= self.parameters['min_histogram']:
            signal_type = "bullish" if crossed_above else "bearish"
            direction = "above" if crossed_above else "below"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': macd_curr,
                'baseline': signal_curr,
                'deviation': abs(histogram_curr),
                'details': (
                    f"MACD {signal_type} crossover: MACD crossed {direction} signal line "
                    f"(histogram: {histogram_curr:.2f})"
                ),
                'metrics': {
                    'macd': macd_curr,
                    'signal': signal_curr,
                    'histogram': histogram_curr,
                    'signal_type': signal_type,
                    'current_price': current_row['Close']
                }
            }

        return None


class VolatilitySpike Rule(BaseAnomalyRule):
    """
    Detects sudden increases in volatility

    Uses ATR (Average True Range) to measure volatility
    """

    def __init__(self):
        super().__init__(
            name="Volatility Spike",
            description="Detects sudden increases in price volatility",
            severity=2,
            enabled=True,
            window_size=14,
            cooldown_periods=5,
            parameters={
                'atr_period': 14,
                'volatility_multiplier': 2.0,
                'extreme_multiplier': 3.0
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate volatility spike"""

        if not self.check_cooldown(current_index):
            return None

        atr_period = self.parameters['atr_period']
        if current_index < atr_period + 1:
            return None

        # Calculate True Range for current and historical periods
        window_data = self.get_window_data(data, current_index, atr_period)
        current_row = data.iloc[current_index]

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        def calculate_tr(df, idx):
            high = df['High'].iloc[idx]
            low = df['Low'].iloc[idx]
            prev_close = df['Close'].iloc[idx - 1] if idx > 0 else df['Close'].iloc[idx]

            return max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )

        # Calculate ATR (average true range)
        true_ranges = []
        for i in range(len(window_data)):
            actual_idx = current_index - atr_period + i + 1
            if actual_idx > 0:
                tr = calculate_tr(data, actual_idx)
                true_ranges.append(tr)

        avg_tr = np.mean(true_ranges) if true_ranges else 0
        current_tr = calculate_tr(data, current_index)

        if avg_tr == 0:
            return None

        volatility_ratio = current_tr / avg_tr

        # Check for volatility spike
        if volatility_ratio >= self.parameters['volatility_multiplier']:
            severity = self.severity

            if volatility_ratio >= self.parameters['extreme_multiplier']:
                severity = min(4, severity + 2)

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_tr,
                'baseline': avg_tr,
                'deviation': volatility_ratio,
                'details': (
                    f"Volatility spike: True Range ${current_tr:.2f} "
                    f"({volatility_ratio:.1f}x average of ${avg_tr:.2f})"
                ),
                'metrics': {
                    'true_range': current_tr,
                    'average_tr': avg_tr,
                    'volatility_ratio': volatility_ratio,
                    'high': current_row['High'],
                    'low': current_row['Low'],
                    'close': current_row['Close']
                }
            }

        return None


class PriceRangContractionRule(BaseAnomalyRule):
    """
    Detects price range contraction (consolidation)

    Often precedes large moves - "calm before the storm"
    """

    def __init__(self):
        super().__init__(
            name="Price Range Contraction",
            description="Detects consolidation periods with unusually tight price ranges",
            severity=1,
            enabled=True,
            window_size=20,
            cooldown_periods=10,
            parameters={
                'contraction_threshold': 0.5,  # Range < 50% of average
                'tight_threshold': 0.3  # Very tight range
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate price range contraction"""

        if not self.check_cooldown(current_index):
            return None

        if current_index < self.window_size:
            return None

        window_data = self.get_window_data(data, current_index, window_size)
        current_row = data.iloc[current_index]

        # Calculate price ranges
        window_data['range'] = window_data['High'] - window_data['Low']
        avg_range = window_data['range'].mean()
        current_range = current_row['High'] - current_row['Low']

        if avg_range == 0:
            return None

        range_ratio = current_range / avg_range

        if range_ratio <= self.parameters['contraction_threshold']:
            condition = "consolidation"
            severity = self.severity

            if range_ratio <= self.parameters['tight_threshold']:
                condition = "tight consolidation"
                severity = min(4, severity + 1)

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_range,
                'baseline': avg_range,
                'deviation': range_ratio,
                'details': (
                    f"Price range {condition}: ${current_range:.2f} "
                    f"({range_ratio:.1%} of average ${avg_range:.2f})"
                ),
                'metrics': {
                    'current_range': current_range,
                    'average_range': avg_range,
                    'range_ratio': range_ratio,
                    'high': current_row['High'],
                    'low': current_row['Low']
                }
            }

        return None
