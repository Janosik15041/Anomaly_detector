"""
anomaly.py - Anomaly Detection for Stock Data

Defines the Anomaly object and detection algorithms
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    PRICE_SPIKE = "price_spike"
    PRICE_DROP = "price_drop"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    TREND_REVERSAL = "trend_reversal"
    GAP = "gap"


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Anomaly:
    """
    Represents a detected anomaly in stock data

    Attributes:
        index: Position in the data where anomaly occurs
        timestamp: Datetime of the anomaly
        type: Type of anomaly detected
        severity: Severity level (1-4)
        value: The actual value that triggered the anomaly
        baseline: The expected/normal value
        deviation: How much it deviates from normal (in standard deviations)
        description: Human-readable description
        metrics: Additional metrics related to the anomaly
    """
    index: int
    timestamp: pd.Timestamp
    type: AnomalyType
    severity: AnomalySeverity
    value: float
    baseline: float
    deviation: float
    description: str
    metrics: Dict[str, float]

    def to_dict(self) -> dict:
        """Convert anomaly to dictionary for JSON serialization"""
        return {
            'index': self.index,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else None,
            'type': self.type.value,
            'severity': self.severity.value,
            'value': float(self.value),
            'baseline': float(self.baseline),
            'deviation': float(self.deviation),
            'description': self.description,
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in self.metrics.items()}
        }


class AnomalyDetector:
    """
    Detects anomalies in streaming stock data using statistical methods
    """

    def __init__(self,
                 window_size: int = 20,
                 z_threshold: float = 3.0,
                 volume_threshold: float = 2.5,
                 volatility_threshold: float = 2.0):
        """
        Initialize the anomaly detector

        Args:
            window_size: Size of rolling window for baseline calculations
            z_threshold: Z-score threshold for price anomalies
            volume_threshold: Threshold multiplier for volume spikes
            volatility_threshold: Threshold for volatility anomalies
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.volume_threshold = volume_threshold
        self.volatility_threshold = volatility_threshold
        self.detected_anomalies: List[Anomaly] = []

    def detect(self, data: pd.DataFrame, current_index: int) -> Optional[Anomaly]:
        """
        Detect anomalies in the current data point

        Args:
            data: Full DataFrame with OHLCV data
            current_index: Index of current data point to check

        Returns:
            Anomaly object if detected, None otherwise
        """
        if current_index < self.window_size:
            return None  # Need enough data for baseline

        # Get window for baseline calculation
        window_start = max(0, current_index - self.window_size)
        window_data = data.iloc[window_start:current_index]
        current_row = data.iloc[current_index]

        # Check for different types of anomalies
        anomaly = None

        # 1. Price spike/drop detection
        anomaly = self._detect_price_anomaly(window_data, current_row, current_index)
        if anomaly:
            return anomaly

        # 2. Volume spike detection
        anomaly = self._detect_volume_anomaly(window_data, current_row, current_index)
        if anomaly:
            return anomaly

        # 3. Volatility spike detection
        anomaly = self._detect_volatility_anomaly(window_data, current_row, current_index)
        if anomaly:
            return anomaly

        # 4. Gap detection
        if current_index > 0:
            anomaly = self._detect_gap(data.iloc[current_index - 1], current_row, current_index)
            if anomaly:
                return anomaly

        return None

    def _detect_price_anomaly(self, window_data: pd.DataFrame,
                             current_row: pd.Series,
                             current_index: int) -> Optional[Anomaly]:
        """Detect unusual price movements using z-score"""
        baseline_mean = window_data['Close'].mean()
        baseline_std = window_data['Close'].std()

        if baseline_std == 0:
            return None

        current_price = current_row['Close']
        z_score = (current_price - baseline_mean) / baseline_std

        if abs(z_score) > self.z_threshold:
            is_spike = z_score > 0
            severity = self._calculate_severity(abs(z_score), self.z_threshold)

            return Anomaly(
                index=current_index,
                timestamp=current_row['Datetime'],
                type=AnomalyType.PRICE_SPIKE if is_spike else AnomalyType.PRICE_DROP,
                severity=severity,
                value=current_price,
                baseline=baseline_mean,
                deviation=abs(z_score),
                description=f"{'Price spike' if is_spike else 'Price drop'} detected: "
                           f"${current_price:.2f} (baseline: ${baseline_mean:.2f}, "
                           f"{z_score:.2f} std devs)",
                metrics={
                    'z_score': z_score,
                    'percent_change': ((current_price - baseline_mean) / baseline_mean) * 100,
                    'std_dev': baseline_std
                }
            )

        return None

    def _detect_volume_anomaly(self, window_data: pd.DataFrame,
                               current_row: pd.Series,
                               current_index: int) -> Optional[Anomaly]:
        """Detect unusual volume spikes"""
        baseline_volume = window_data['Volume'].mean()
        current_volume = current_row['Volume']

        if baseline_volume == 0:
            return None

        volume_ratio = current_volume / baseline_volume

        if volume_ratio > self.volume_threshold:
            severity = self._calculate_severity(volume_ratio, self.volume_threshold)

            return Anomaly(
                index=current_index,
                timestamp=current_row['Datetime'],
                type=AnomalyType.VOLUME_SPIKE,
                severity=severity,
                value=current_volume,
                baseline=baseline_volume,
                deviation=volume_ratio,
                description=f"Volume spike detected: {current_volume:,.0f} "
                           f"({volume_ratio:.1f}x baseline of {baseline_volume:,.0f})",
                metrics={
                    'volume_ratio': volume_ratio,
                    'baseline_volume': baseline_volume
                }
            )

        return None

    def _detect_volatility_anomaly(self, window_data: pd.DataFrame,
                                   current_row: pd.Series,
                                   current_index: int) -> Optional[Anomaly]:
        """Detect unusual volatility (price range)"""
        # Calculate historical volatility (high-low range)
        window_data['range'] = window_data['High'] - window_data['Low']
        baseline_volatility = window_data['range'].mean()
        baseline_std = window_data['range'].std()

        current_range = current_row['High'] - current_row['Low']

        if baseline_std == 0:
            return None

        z_score = (current_range - baseline_volatility) / baseline_std

        if z_score > self.volatility_threshold:
            severity = self._calculate_severity(z_score, self.volatility_threshold)

            return Anomaly(
                index=current_index,
                timestamp=current_row['Datetime'],
                type=AnomalyType.VOLATILITY_SPIKE,
                severity=severity,
                value=current_range,
                baseline=baseline_volatility,
                deviation=z_score,
                description=f"Volatility spike detected: ${current_range:.2f} range "
                           f"(baseline: ${baseline_volatility:.2f})",
                metrics={
                    'z_score': z_score,
                    'high': current_row['High'],
                    'low': current_row['Low']
                }
            )

        return None

    def _detect_gap(self, previous_row: pd.Series,
                   current_row: pd.Series,
                   current_index: int) -> Optional[Anomaly]:
        """Detect price gaps (opening significantly different from previous close)"""
        prev_close = previous_row['Close']
        current_open = current_row['Open']

        gap_percent = abs((current_open - prev_close) / prev_close) * 100

        # Consider it a gap if > 2% difference
        if gap_percent > 2.0:
            is_gap_up = current_open > prev_close
            severity = self._calculate_severity(gap_percent, 2.0)

            return Anomaly(
                index=current_index,
                timestamp=current_row['Datetime'],
                type=AnomalyType.GAP,
                severity=severity,
                value=current_open,
                baseline=prev_close,
                deviation=gap_percent,
                description=f"{'Gap up' if is_gap_up else 'Gap down'} detected: "
                           f"{gap_percent:.1f}% difference from previous close",
                metrics={
                    'gap_percent': gap_percent,
                    'previous_close': prev_close,
                    'current_open': current_open,
                    'direction': 'up' if is_gap_up else 'down'
                }
            )

        return None

    def _calculate_severity(self, value: float, threshold: float) -> AnomalySeverity:
        """Calculate severity based on how much value exceeds threshold"""
        ratio = value / threshold

        if ratio > 3:
            return AnomalySeverity.CRITICAL
        elif ratio > 2:
            return AnomalySeverity.HIGH
        elif ratio > 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def add_anomaly(self, anomaly: Anomaly):
        """Add detected anomaly to the list"""
        self.detected_anomalies.append(anomaly)

    def get_anomalies(self) -> List[Anomaly]:
        """Get all detected anomalies"""
        return self.detected_anomalies

    def clear_anomalies(self):
        """Clear all detected anomalies"""
        self.detected_anomalies.clear()

    def get_anomalies_dict(self) -> List[dict]:
        """Get all anomalies as dictionaries for JSON serialization"""
        return [a.to_dict() for a in self.detected_anomalies]
