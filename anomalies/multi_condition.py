"""
Multi-Condition Anomaly Rules

Complex rules that combine multiple conditions for more accurate detection
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_rule import BaseAnomalyRule


class BreakoutWithVolumeRule(BaseAnomalyRule):
    """
    Detects price breakouts confirmed by volume

    More reliable than price breakouts alone - volume confirmation
    reduces false signals
    """

    def __init__(self):
        super().__init__(
            name="Breakout with Volume Confirmation",
            description="Detects price breakouts confirmed by above-average volume",
            severity=3,
            enabled=True,
            window_size=20,
            cooldown_periods=10,
            parameters={
                'price_std_threshold': 2.0,
                'volume_multiplier': 1.5,
                'min_price_change': 3.0  # 3% minimum price change
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate breakout with volume"""

        if not self.check_cooldown(current_index):
            return None

        if current_index < self.window_size:
            return None

        window_data = self.get_window_data(data, current_index, window_size)
        current_row = data.iloc[current_index]

        # Check price breakout
        price_zscore = self.calculate_zscore(
            current_row['Close'],
            window_data['Close']
        )

        avg_price = window_data['Close'].mean()
        price_change_pct = self.calculate_percent_change(
            current_row['Close'],
            avg_price
        )

        # Check volume confirmation
        avg_volume = window_data['Volume'].mean()
        volume_ratio = current_row['Volume'] / avg_volume if avg_volume > 0 else 0

        # Both conditions must be met
        price_breakout = (
            abs(price_zscore) >= self.parameters['price_std_threshold'] and
            abs(price_change_pct) >= self.parameters['min_price_change']
        )
        volume_confirmed = volume_ratio >= self.parameters['volume_multiplier']

        if price_breakout and volume_confirmed:
            direction = "bullish" if price_zscore > 0 else "bearish"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_row['Close'],
                'baseline': avg_price,
                'deviation': abs(price_zscore),
                'details': (
                    f"Volume-confirmed {direction} breakout: "
                    f"Price ${current_row['Close']:.2f} ({price_change_pct:+.1f}%) "
                    f"with {volume_ratio:.1f}x volume"
                ),
                'metrics': {
                    'price_zscore': price_zscore,
                    'price_change_pct': price_change_pct,
                    'volume_ratio': volume_ratio,
                    'avg_volume': avg_volume,
                    'current_volume': current_row['Volume']
                }
            }

        return None


class ReversalPatternRule(BaseAnomalyRule):
    """
    Detects potential trend reversal patterns

    Combines:
    - Extreme price extension
    - Decreasing momentum
    - Volume characteristics
    """

    def __init__(self):
        super().__init__(
            name="Reversal Pattern",
            description="Detects potential trend reversals using multiple indicators",
            severity=3,
            enabled=True,
            window_size=20,
            cooldown_periods=15,
            parameters={
                'price_extension_threshold': 2.5,  # Z-score
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'volume_fade_threshold': 0.8  # Volume declining to 80% of average
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate reversal pattern"""

        if not self.check_cooldown(current_index):
            return None

        if current_index < self.window_size + 14:  # Need RSI period
            return None

        window_data = self.get_window_data(data, current_index, window_size)
        current_row = data.iloc[current_index]

        # 1. Check price extension
        price_zscore = self.calculate_zscore(
            current_row['Close'],
            window_data['Close']
        )

        # 2. Calculate RSI
        rsi = self.calculate_rsi(data['Close'].iloc[:current_index + 1], period=14)

        # 3. Check volume
        avg_volume = window_data['Volume'].mean()
        volume_ratio = current_row['Volume'] / avg_volume if avg_volume > 0 else 0

        # Detect bullish reversal (bottom)
        if (price_zscore <= -self.parameters['price_extension_threshold'] and
            rsi <= self.parameters['rsi_oversold']):

            reversal_type = "bullish reversal (potential bottom)"
            confidence = "high" if volume_ratio < self.parameters['volume_fade_threshold'] else "moderate"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_row['Close'],
                'baseline': window_data['Close'].mean(),
                'deviation': abs(price_zscore),
                'details': (
                    f"{reversal_type.capitalize()} - "
                    f"Price oversold (RSI: {rsi:.1f}), "
                    f"{confidence} confidence"
                ),
                'metrics': {
                    'reversal_type': 'bullish',
                    'price_zscore': price_zscore,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'confidence': confidence
                }
            }

        # Detect bearish reversal (top)
        elif (price_zscore >= self.parameters['price_extension_threshold'] and
              rsi >= self.parameters['rsi_overbought']):

            reversal_type = "bearish reversal (potential top)"
            confidence = "high" if volume_ratio < self.parameters['volume_fade_threshold'] else "moderate"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_row['Close'],
                'baseline': window_data['Close'].mean(),
                'deviation': abs(price_zscore),
                'details': (
                    f"{reversal_type.capitalize()} - "
                    f"Price overbought (RSI: {rsi:.1f}), "
                    f"{confidence} confidence"
                ),
                'metrics': {
                    'reversal_type': 'bearish',
                    'price_zscore': price_zscore,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'confidence': confidence
                }
            }

        return None


class FlashCrashRule(BaseAnomalyRule):
    """
    Detects flash crash patterns

    Characteristics:
    - Sudden sharp drop
    - High volume
    - Quick recovery (partial or full)
    """

    def __init__(self):
        super().__init__(
            name="Flash Crash",
            description="Detects flash crash patterns with rapid price drops and recovery",
            severity=4,
            enabled=True,
            window_size=10,
            cooldown_periods=20,
            parameters={
                'crash_threshold': -5.0,  # 5% drop
                'recovery_threshold': 50,  # Recover at least 50% of loss
                'volume_spike': 3.0,
                'time_window': 3  # Look for recovery within 3 periods
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate flash crash pattern"""

        if not self.check_cooldown(current_index):
            return None

        recovery_window = self.parameters['time_window']
        if current_index < self.window_size or current_index < recovery_window:
            return None

        current_row = data.iloc[current_index]
        window_data = self.get_window_data(data, current_index, window_size)

        # Check for sharp drop in recent periods
        for i in range(1, recovery_window + 1):
            if current_index - i < 0:
                continue

            crash_row = data.iloc[current_index - i]
            pre_crash_price = data.iloc[current_index - i - 1]['Close'] if current_index - i - 1 >= 0 else crash_row['Open']

            # Calculate drop
            drop_pct = self.calculate_percent_change(crash_row['Low'], pre_crash_price)

            if drop_pct <= self.parameters['crash_threshold']:
                # Found a crash, check for recovery
                recovery_pct = self.calculate_percent_change(
                    current_row['Close'],
                    crash_row['Low']
                )

                # Calculate how much of the drop was recovered
                total_drop = abs(crash_row['Low'] - pre_crash_price)
                recovered_amount = current_row['Close'] - crash_row['Low']
                recovery_ratio = (recovered_amount / total_drop * 100) if total_drop > 0 else 0

                # Check volume
                avg_volume = window_data['Volume'].mean()
                crash_volume = crash_row['Volume']
                volume_ratio = crash_volume / avg_volume if avg_volume > 0 else 0

                # Confirm flash crash pattern
                if (recovery_ratio >= self.parameters['recovery_threshold'] and
                    volume_ratio >= self.parameters['volume_spike']):

                    self.reset_cooldown(current_index)

                    return {
                        'triggered': True,
                        'value': crash_row['Low'],
                        'baseline': pre_crash_price,
                        'deviation': abs(drop_pct),
                        'details': (
                            f"Flash crash detected: {abs(drop_pct):.1f}% drop "
                            f"with {recovery_ratio:.0f}% recovery "
                            f"({volume_ratio:.1f}x volume)"
                        ),
                        'metrics': {
                            'crash_price': crash_row['Low'],
                            'pre_crash_price': pre_crash_price,
                            'current_price': current_row['Close'],
                            'drop_percent': drop_pct,
                            'recovery_percent': recovery_ratio,
                            'volume_ratio': volume_ratio,
                            'periods_ago': i
                        }
                    }

        return None


class SqueezBreakoutRule(BaseAnomalyRule):
    """
    Detects squeeze breakouts (Bollinger Band squeeze followed by expansion)

    When volatility contracts (squeeze), it often precedes large moves
    """

    def __init__(self):
        super().__init__(
            name="Squeeze Breakout",
            description="Detects volatility squeeze followed by breakout",
            severity=3,
            enabled=True,
            window_size=20,
            cooldown_periods=20,
            parameters={
                'squeeze_threshold': 0.05,  # BB width relative to price
                'breakout_threshold': 2.0,  # Z-score for breakout
                'volume_confirmation': 1.5
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate squeeze breakout"""

        if not self.check_cooldown(current_index):
            return None

        ws = window_size or self.window_size
        if current_index < ws + 5:
            return None

        # Check for recent squeeze
        squeeze_detected = False
        squeeze_idx = None

        for i in range(1, 6):  # Check last 5 periods
            check_idx = current_index - i
            if check_idx < ws:
                continue

            check_window = data.iloc[check_idx - ws:check_idx]
            upper, middle, lower = self.calculate_bollinger_bands(
                check_window['Close'],
                window=ws,
                num_std=2.0
            )

            # Calculate band width relative to price
            band_width = (upper - lower) / middle if middle > 0 else 0

            if band_width <= self.parameters['squeeze_threshold']:
                squeeze_detected = True
                squeeze_idx = check_idx
                break

        if not squeeze_detected:
            return None

        # Now check for breakout
        current_row = data.iloc[current_index]
        window_data = self.get_window_data(data, current_index, ws)

        price_zscore = self.calculate_zscore(
            current_row['Close'],
            window_data['Close']
        )

        avg_volume = window_data['Volume'].mean()
        volume_ratio = current_row['Volume'] / avg_volume if avg_volume > 0 else 0

        if (abs(price_zscore) >= self.parameters['breakout_threshold'] and
            volume_ratio >= self.parameters['volume_confirmation']):

            direction = "bullish" if price_zscore > 0 else "bearish"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_row['Close'],
                'baseline': window_data['Close'].mean(),
                'deviation': abs(price_zscore),
                'details': (
                    f"Squeeze breakout ({direction}): "
                    f"Volatility squeeze followed by {abs(price_zscore):.1f}σ move "
                    f"with {volume_ratio:.1f}x volume"
                ),
                'metrics': {
                    'price_zscore': price_zscore,
                    'volume_ratio': volume_ratio,
                    'squeeze_periods_ago': current_index - squeeze_idx,
                    'direction': direction
                }
            }

        return None
