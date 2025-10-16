"""
Base Anomaly Rule - Foundation for custom anomaly detection rules

Users can create custom rules by inheriting from BaseAnomalyRule
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from dataclasses import dataclass


class RuleOperator(Enum):
    """Operators for rule conditions"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN_RANGE = "in_range"
    OUT_OF_RANGE = "out_of_range"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    PERCENT_CHANGE = "percent_change"
    STANDARD_DEV = "std_dev"


@dataclass
class RuleCondition:
    """
    Represents a single condition in an anomaly rule

    Examples:
        - Price > $100
        - Volume > 2x average
        - RSI < 30
        - Price crosses above moving average
    """
    metric: str  # e.g., 'Close', 'Volume', 'RSI'
    operator: RuleOperator
    threshold: float
    description: str


class BaseAnomalyRule(ABC):
    """
    Base class for all custom anomaly detection rules

    Users create custom rules by:
    1. Inheriting from this class
    2. Setting metadata (name, description, severity)
    3. Implementing the evaluate() method with their custom logic

    Example:
        class MyCustomRule(BaseAnomalyRule):
            def __init__(self):
                super().__init__(
                    name="Custom Price Spike",
                    description="Detects when price increases >5% with high volume",
                    severity=3,
                    enabled=True
                )

            def evaluate(self, data, current_index, window_size):
                # Custom logic here
                pass
    """

    def __init__(self,
                 name: str,
                 description: str,
                 severity: int = 2,
                 enabled: bool = True,
                 window_size: int = 20,
                 cooldown_periods: int = 5,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly rule

        Args:
            name: Rule name
            description: Human-readable description
            severity: Severity level (1-4)
            enabled: Whether rule is active
            window_size: Default window size for calculations
            cooldown_periods: Number of periods to wait before detecting same anomaly again
            parameters: Custom parameters for the rule
        """
        self.name = name
        self.description = description
        self.severity = severity
        self.enabled = enabled
        self.window_size = window_size
        self.cooldown_periods = cooldown_periods
        self.parameters = parameters or {}
        self.last_triggered_index = -cooldown_periods

    @abstractmethod
    def evaluate(self,
                data: pd.DataFrame,
                current_index: int,
                window_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Evaluate the rule on current data point

        Args:
            data: Full DataFrame with stock data
            current_index: Index of current point to evaluate
            window_size: Window size (uses default if not provided)

        Returns:
            Dictionary with anomaly details if detected, None otherwise
            Format: {
                'triggered': bool,
                'value': float,
                'baseline': float,
                'deviation': float,
                'details': str,
                'metrics': dict
            }
        """
        pass

    def check_cooldown(self, current_index: int) -> bool:
        """Check if cooldown period has passed"""
        return (current_index - self.last_triggered_index) >= self.cooldown_periods

    def reset_cooldown(self, current_index: int):
        """Reset cooldown timer"""
        self.last_triggered_index = current_index

    def get_window_data(self,
                       data: pd.DataFrame,
                       current_index: int,
                       window_size: Optional[int] = None) -> pd.DataFrame:
        """Get window of historical data for baseline calculations"""
        ws = window_size or self.window_size
        start_idx = max(0, current_index - ws)
        return data.iloc[start_idx:current_index]

    def calculate_zscore(self,
                        value: float,
                        series: pd.Series) -> float:
        """Calculate z-score of value compared to series"""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return 0
        return (value - mean) / std

    def calculate_percent_change(self,
                                 current: float,
                                 baseline: float) -> float:
        """Calculate percent change"""
        if baseline == 0:
            return 0
        return ((current - baseline) / baseline) * 100

    def moving_average(self,
                      series: pd.Series,
                      window: int) -> float:
        """Calculate moving average"""
        return series.tail(window).mean()

    def detect_crossover(self,
                        series: pd.Series,
                        threshold: float,
                        direction: str = 'above') -> bool:
        """
        Detect if series crosses above/below threshold

        Args:
            series: Pandas series (last 2 values checked)
            threshold: Threshold value
            direction: 'above' or 'below'
        """
        if len(series) < 2:
            return False

        prev_val = series.iloc[-2]
        curr_val = series.iloc[-1]

        if direction == 'above':
            return prev_val <= threshold < curr_val
        else:  # below
            return prev_val >= threshold > curr_val

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def calculate_bollinger_bands(self,
                                  prices: pd.Series,
                                  window: int = 20,
                                  num_std: float = 2.0) -> tuple:
        """
        Calculate Bollinger Bands

        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=window).mean().iloc[-1]
        std = prices.rolling(window=window).std().iloc[-1]
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return upper, middle, lower

    def calculate_macd(self,
                      prices: pd.Series,
                      fast: int = 12,
                      slow: int = 26,
                      signal: int = 9) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Returns:
            (macd_line, signal_line, histogram)
        """
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (macd_line.iloc[-1] if len(macd_line) > 0 else 0,
                signal_line.iloc[-1] if len(signal_line) > 0 else 0,
                histogram.iloc[-1] if len(histogram) > 0 else 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'severity': self.severity,
            'enabled': self.enabled,
            'window_size': self.window_size,
            'cooldown_periods': self.cooldown_periods,
            'parameters': self.parameters
        }
