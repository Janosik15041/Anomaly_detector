"""
Price Pattern Detection Rules

Detects various price patterns and movements including:
- Sharp price spikes/drops
- Bollinger Band breakouts
- Moving average crossovers
- Support/resistance breaks
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_rule import BaseAnomalyRule


class SharpPriceMovementRule(BaseAnomalyRule):
    """
    Detects sharp price movements using standard deviations

    More sophisticated than simple threshold - uses statistical analysis
    """

    def __init__(self):
        super().__init__(
            name="Sharp Price Movement",
            description="Detects statistically significant price movements",
            severity=3,
            enabled=True,
            window_size=20,
            cooldown_periods=3,
            parameters={
                'z_score_threshold': 2.5,
                'critical_z_score': 4.0,
                'min_percent_change': 2.0  # Minimum 2% change
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate sharp price movement"""

        if not self.check_cooldown(current_index):
            return None

        if current_index < self.window_size:
            return None

        window_data = self.get_window_data(data, current_index, window_size)
        current_row = data.iloc[current_index]

        # Calculate z-score
        z_score = self.calculate_zscore(
            current_row['Close'],
            window_data['Close']
        )

        # Calculate percent change
        avg_price = window_data['Close'].mean()
        pct_change = self.calculate_percent_change(
            current_row['Close'],
            avg_price
        )

        # Check if significant
        if (abs(z_score) >= self.parameters['z_score_threshold'] and
            abs(pct_change) >= self.parameters['min_percent_change']):

            # Determine severity
            severity = self.severity
            if abs(z_score) >= self.parameters['critical_z_score']:
                severity = 4

            direction = "spike" if z_score > 0 else "drop"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_row['Close'],
                'baseline': avg_price,
                'deviation': abs(z_score),
                'details': (
                    f"Sharp price {direction}: ${current_row['Close']:.2f} "
                    f"({pct_change:+.1f}%, {z_score:.1f}σ from baseline)"
                ),
                'metrics': {
                    'z_score': z_score,
                    'percent_change': pct_change,
                    'baseline_price': avg_price,
                    'std_dev': window_data['Close'].std()
                }
            }

        return None


class BollingerBandBreakoutRule(BaseAnomalyRule):
    """
    Detects when price breaks out of Bollinger Bands

    Bollinger Bands are volatility bands placed above and below a moving average
    """

    def __init__(self):
        super().__init__(
            name="Bollinger Band Breakout",
            description="Detects price breakouts beyond Bollinger Bands",
            severity=2,
            enabled=True,
            window_size=20,
            cooldown_periods=5,
            parameters={
                'num_std': 2.0,  # Standard deviations for bands
                'extreme_std': 3.0  # Extreme breakout
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate Bollinger Band breakout"""

        if not self.check_cooldown(current_index):
            return None

        ws = window_size or self.window_size
        if current_index < ws:
            return None

        window_data = self.get_window_data(data, current_index, window_size)
        current_row = data.iloc[current_index]

        # Calculate Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(
            window_data['Close'],
            window=ws,
            num_std=self.parameters['num_std']
        )

        current_price = current_row['Close']

        # Check for breakout
        if current_price > upper or current_price < lower:
            direction = "above" if current_price > upper else "below"
            band = upper if current_price > upper else lower

            # Calculate how far outside the band
            distance_from_band = abs(current_price - band)
            pct_outside = (distance_from_band / middle) * 100

            # Check for extreme breakout
            if direction == "above":
                extreme_upper = middle + (self.parameters['extreme_std'] * window_data['Close'].std())
                is_extreme = current_price > extreme_upper
            else:
                extreme_lower = middle - (self.parameters['extreme_std'] * window_data['Close'].std())
                is_extreme = current_price < extreme_lower

            severity = self.severity + 1 if is_extreme else self.severity

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_price,
                'baseline': middle,
                'deviation': distance_from_band,
                'details': (
                    f"Bollinger Band breakout: Price ${current_price:.2f} "
                    f"broke {direction} band at ${band:.2f} "
                    f"({pct_outside:.1f}% outside)"
                ),
                'metrics': {
                    'upper_band': upper,
                    'middle_band': middle,
                    'lower_band': lower,
                    'distance_from_band': distance_from_band,
                    'percent_outside': pct_outside,
                    'is_extreme': is_extreme
                }
            }

        return None


class MovingAverageCrossoverRule(BaseAnomalyRule):
    """
    Detects golden cross and death cross patterns

    Golden Cross: Fast MA crosses above slow MA (bullish)
    Death Cross: Fast MA crosses below slow MA (bearish)
    """

    def __init__(self):
        super().__init__(
            name="Moving Average Crossover",
            description="Detects golden cross and death cross patterns",
            severity=2,
            enabled=True,
            window_size=50,
            cooldown_periods=10,
            parameters={
                'fast_period': 20,
                'slow_period': 50
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate moving average crossover"""

        if not self.check_cooldown(current_index):
            return None

        slow_period = self.parameters['slow_period']
        if current_index < slow_period + 1:
            return None

        # Calculate moving averages
        fast_period = self.parameters['fast_period']

        fast_ma_prev = data['Close'].iloc[current_index - fast_period:current_index - 1].mean()
        fast_ma_curr = data['Close'].iloc[current_index - fast_period + 1:current_index + 1].mean()

        slow_ma_prev = data['Close'].iloc[current_index - slow_period:current_index - 1].mean()
        slow_ma_curr = data['Close'].iloc[current_index - slow_period + 1:current_index + 1].mean()

        current_row = data.iloc[current_index]

        # Detect crossover
        crossed_above = fast_ma_prev <= slow_ma_prev and fast_ma_curr > slow_ma_curr
        crossed_below = fast_ma_prev >= slow_ma_prev and fast_ma_curr < slow_ma_curr

        if crossed_above or crossed_below:
            cross_type = "Golden Cross" if crossed_above else "Death Cross"
            direction = "bullish" if crossed_above else "bearish"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_row['Close'],
                'baseline': slow_ma_curr,
                'deviation': abs(fast_ma_curr - slow_ma_curr),
                'details': (
                    f"{cross_type} detected: {fast_period}-period MA crossed "
                    f"{'above' if crossed_above else 'below'} {slow_period}-period MA "
                    f"({direction} signal)"
                ),
                'metrics': {
                    'fast_ma': fast_ma_curr,
                    'slow_ma': slow_ma_curr,
                    'cross_type': cross_type,
                    'current_price': current_row['Close']
                }
            }

        return None


class GapDetectionRule(BaseAnomalyRule):
    """
    Detects price gaps (opening price significantly different from previous close)

    Common after earnings reports, news events, or overnight market moves
    """

    def __init__(self):
        super().__init__(
            name="Price Gap",
            description="Detects significant gaps between open and previous close",
            severity=2,
            enabled=True,
            window_size=5,
            cooldown_periods=1,
            parameters={
                'gap_threshold': 2.0,  # 2% gap
                'large_gap_threshold': 5.0  # 5% large gap
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate price gap"""

        if current_index < 1:
            return None

        if not self.check_cooldown(current_index):
            return None

        previous_row = data.iloc[current_index - 1]
        current_row = data.iloc[current_index]

        prev_close = previous_row['Close']
        current_open = current_row['Open']

        gap_pct = self.calculate_percent_change(current_open, prev_close)

        if abs(gap_pct) >= self.parameters['gap_threshold']:
            direction = "up" if gap_pct > 0 else "down"

            # Determine severity
            severity = self.severity
            if abs(gap_pct) >= self.parameters['large_gap_threshold']:
                severity = min(4, severity + 2)

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_open,
                'baseline': prev_close,
                'deviation': abs(gap_pct),
                'details': (
                    f"Gap {direction}: {abs(gap_pct):.1f}% gap "
                    f"(open ${current_open:.2f} vs prev close ${prev_close:.2f})"
                ),
                'metrics': {
                    'gap_percent': gap_pct,
                    'previous_close': prev_close,
                    'current_open': current_open,
                    'direction': direction
                }
            }

        return None
