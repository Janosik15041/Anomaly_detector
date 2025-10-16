"""
Volume Surge Detection Rule

Detects unusual volume spikes that may indicate institutional trading,
news events, or market manipulation
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_rule import BaseAnomalyRule


class VolumeSurgeRule(BaseAnomalyRule):
    """
    Detects volume surges that exceed historical averages

    Triggers when:
    - Volume is significantly higher than moving average
    - Can combine with price movement for stronger signals
    """

    def __init__(self):
        super().__init__(
            name="Volume Surge",
            description="Detects unusual volume spikes compared to historical average",
            severity=2,
            enabled=True,
            window_size=20,
            cooldown_periods=3,
            parameters={
                'volume_multiplier': 2.5,  # Trigger at 2.5x average volume
                'extreme_multiplier': 5.0,  # Extreme surge threshold
                'consider_price_movement': True
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate volume surge"""

        # Check cooldown
        if not self.check_cooldown(current_index):
            return None

        # Need enough data
        if current_index < self.window_size:
            return None

        # Get window data
        window_data = self.get_window_data(data, current_index, window_size)
        current_row = data.iloc[current_index]

        # Calculate baseline volume
        avg_volume = window_data['Volume'].mean()
        std_volume = window_data['Volume'].std()
        current_volume = current_row['Volume']

        if avg_volume == 0:
            return None

        volume_ratio = current_volume / avg_volume

        # Check if volume exceeds threshold
        if volume_ratio >= self.parameters['volume_multiplier']:

            # Calculate price movement if configured
            price_change_pct = 0
            if self.parameters['consider_price_movement']:
                price_change_pct = self.calculate_percent_change(
                    current_row['Close'],
                    window_data['Close'].mean()
                )

            # Determine severity
            severity = self.severity
            if volume_ratio >= self.parameters['extreme_multiplier']:
                severity = min(4, severity + 2)
            elif volume_ratio >= self.parameters['volume_multiplier'] * 2:
                severity = min(4, severity + 1)

            # Build description
            description = (
                f"Volume surge: {current_volume:,.0f} "
                f"({volume_ratio:.1f}x average of {avg_volume:,.0f})"
            )

            if abs(price_change_pct) > 2:
                direction = "up" if price_change_pct > 0 else "down"
                description += f" with {abs(price_change_pct):.1f}% price move {direction}"

            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_volume,
                'baseline': avg_volume,
                'deviation': volume_ratio,
                'details': description,
                'metrics': {
                    'volume_ratio': volume_ratio,
                    'avg_volume': avg_volume,
                    'std_volume': std_volume,
                    'price_change_pct': price_change_pct,
                    'current_price': current_row['Close']
                }
            }

        return None


class VolumeDropRule(BaseAnomalyRule):
    """
    Detects unusually low volume (potential liquidity issues)

    Triggers when:
    - Volume falls significantly below historical averages
    """

    def __init__(self):
        super().__init__(
            name="Volume Drop",
            description="Detects unusually low volume indicating potential liquidity issues",
            severity=1,
            enabled=True,
            window_size=20,
            cooldown_periods=5,
            parameters={
                'volume_threshold': 0.3  # Trigger when volume < 30% of average
            }
        )

    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Evaluate volume drop"""

        if not self.check_cooldown(current_index):
            return None

        if current_index < self.window_size:
            return None

        window_data = self.get_window_data(data, current_index, window_size)
        current_row = data.iloc[current_index]

        avg_volume = window_data['Volume'].mean()
        current_volume = current_row['Volume']

        if avg_volume == 0:
            return None

        volume_ratio = current_volume / avg_volume

        if volume_ratio <= self.parameters['volume_threshold']:
            self.reset_cooldown(current_index)

            return {
                'triggered': True,
                'value': current_volume,
                'baseline': avg_volume,
                'deviation': volume_ratio,
                'details': (
                    f"Volume drop: {current_volume:,.0f} "
                    f"({volume_ratio:.1%} of average {avg_volume:,.0f})"
                ),
                'metrics': {
                    'volume_ratio': volume_ratio,
                    'avg_volume': avg_volume,
                    'current_price': current_row['Close']
                }
            }

        return None
