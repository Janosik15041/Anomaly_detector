"""
Test suite for the Anomaly Detector system

Run with: python -m pytest tests/test_anomaly_detector.py -v
or: python tests/test_anomaly_detector.py
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.anomaly import AnomalyDetector, AnomalyType, Anomaly, AnomalySeverity
from utils.persistent_random_data import SyntheticStockGenerator


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = AnomalyDetector(
            window_size=20,
            z_threshold=3.0,
            volume_threshold=2.5,
            volatility_threshold=2.0
        )

    def create_sample_data(self, rows=100):
        """Create sample stock data for testing"""
        dates = [datetime.now() + timedelta(hours=i) for i in range(rows)]
        data = pd.DataFrame({
            'Datetime': dates,
            'Open': np.random.uniform(100, 110, rows),
            'High': np.random.uniform(110, 120, rows),
            'Low': np.random.uniform(90, 100, rows),
            'Close': np.random.uniform(100, 110, rows),
            'Volume': np.random.uniform(1000000, 2000000, rows)
        })
        return data

    def test_detector_initialization(self):
        """Test that detector initializes with correct parameters"""
        self.assertEqual(self.detector.window_size, 20)
        self.assertEqual(self.detector.z_threshold, 3.0)
        self.assertEqual(self.detector.volume_threshold, 2.5)
        self.assertEqual(self.detector.volatility_threshold, 2.0)
        self.assertEqual(len(self.detector.detected_anomalies), 0)

    def test_price_spike_detection(self):
        """Test detection of price spikes"""
        data = self.create_sample_data(50)
        # Create a price spike at index 30
        data.loc[30, 'Close'] = 200.0  # Much higher than normal

        anomaly = self.detector.detect(data, 30)

        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.type, AnomalyType.PRICE_SPIKE)
        self.assertGreater(anomaly.deviation, self.detector.z_threshold)

    def test_price_drop_detection(self):
        """Test detection of price drops"""
        data = self.create_sample_data(50)
        # Create a price drop at index 30
        data.loc[30, 'Close'] = 20.0  # Much lower than normal

        anomaly = self.detector.detect(data, 30)

        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.type, AnomalyType.PRICE_DROP)

    def test_volume_spike_detection(self):
        """Test detection of volume spikes"""
        data = self.create_sample_data(50)
        # Create a volume spike at index 30
        data.loc[30, 'Volume'] = 10000000  # Much higher than normal

        anomaly = self.detector.detect(data, 30)

        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.type, AnomalyType.VOLUME_SPIKE)

    def test_volatility_spike_detection(self):
        """Test detection of volatility spikes"""
        data = self.create_sample_data(50)
        # Create a volatility spike at index 30
        data.loc[30, 'High'] = 150.0
        data.loc[30, 'Low'] = 50.0  # Large range

        anomaly = self.detector.detect(data, 30)

        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.type, AnomalyType.VOLATILITY_SPIKE)

    def test_gap_detection(self):
        """Test detection of price gaps"""
        data = self.create_sample_data(50)
        # Create a gap at index 30
        data.loc[29, 'Close'] = 100.0
        data.loc[30, 'Open'] = 110.0  # 10% gap

        anomaly = self.detector.detect(data, 30)

        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.type, AnomalyType.GAP)

    def test_no_anomaly_on_normal_data(self):
        """Test that normal data doesn't trigger anomalies"""
        data = self.create_sample_data(50)
        # Use consistent, normal values
        data['Close'] = 105.0
        data['Volume'] = 1500000
        data['High'] = 106.0
        data['Low'] = 104.0

        anomaly = self.detector.detect(data, 30)

        self.assertIsNone(anomaly)

    def test_insufficient_data(self):
        """Test that detector handles insufficient data gracefully"""
        data = self.create_sample_data(10)  # Less than window_size

        anomaly = self.detector.detect(data, 5)

        self.assertIsNone(anomaly)

    def test_add_and_get_anomalies(self):
        """Test adding and retrieving anomalies"""
        data = self.create_sample_data(50)

        # Detect and add anomaly
        data.loc[30, 'Close'] = 200.0
        anomaly = self.detector.detect(data, 30)
        self.detector.add_anomaly(anomaly)

        # Retrieve anomalies
        anomalies = self.detector.get_anomalies()

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].index, 30)

    def test_clear_anomalies(self):
        """Test clearing anomalies"""
        data = self.create_sample_data(50)
        data.loc[30, 'Close'] = 200.0
        anomaly = self.detector.detect(data, 30)
        self.detector.add_anomaly(anomaly)

        self.assertEqual(len(self.detector.get_anomalies()), 1)

        self.detector.clear_anomalies()

        self.assertEqual(len(self.detector.get_anomalies()), 0)

    def test_anomaly_to_dict(self):
        """Test anomaly serialization to dictionary"""
        data = self.create_sample_data(50)
        data.loc[30, 'Close'] = 200.0
        anomaly = self.detector.detect(data, 30)

        anomaly_dict = anomaly.to_dict()

        self.assertIn('index', anomaly_dict)
        self.assertIn('timestamp', anomaly_dict)
        self.assertIn('type', anomaly_dict)
        self.assertIn('severity', anomaly_dict)
        self.assertIn('value', anomaly_dict)
        self.assertIn('description', anomaly_dict)

    def test_severity_calculation(self):
        """Test severity levels are calculated correctly"""
        # Critical severity (ratio > 3)
        severity = self.detector._calculate_severity(10.0, 3.0)
        self.assertEqual(severity, AnomalySeverity.CRITICAL)

        # High severity (ratio > 2)
        severity = self.detector._calculate_severity(7.0, 3.0)
        self.assertEqual(severity, AnomalySeverity.HIGH)

        # Medium severity (ratio > 1.5)
        severity = self.detector._calculate_severity(5.0, 3.0)
        self.assertEqual(severity, AnomalySeverity.MEDIUM)

        # Low severity
        severity = self.detector._calculate_severity(4.0, 3.0)
        self.assertEqual(severity, AnomalySeverity.LOW)


class TestSyntheticStockGenerator(unittest.TestCase):
    """Test cases for SyntheticStockGenerator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.generator = SyntheticStockGenerator(start_price=100.0)

    def test_generator_initialization(self):
        """Test that generator initializes correctly"""
        self.assertEqual(self.generator.current_price, 100.0)
        self.assertEqual(len(self.generator.generated_candles), 0)

    def test_generate_single_candle(self):
        """Test generating a single candle"""
        candle = self.generator.generate_next_candle()

        self.assertIn('Datetime', candle)
        self.assertIn('Open', candle)
        self.assertIn('High', candle)
        self.assertIn('Low', candle)
        self.assertIn('Close', candle)
        self.assertIn('Volume', candle)

        # Verify OHLC relationships
        self.assertGreaterEqual(candle['High'], candle['Open'])
        self.assertGreaterEqual(candle['High'], candle['Close'])
        self.assertLessEqual(candle['Low'], candle['Open'])
        self.assertLessEqual(candle['Low'], candle['Close'])

    def test_generate_batch(self):
        """Test generating a batch of candles"""
        num_candles = 50
        data = self.generator.generate_batch(num_candles)

        self.assertEqual(len(data), num_candles)
        self.assertEqual(len(self.generator.generated_candles), num_candles)

    def test_get_dataframe(self):
        """Test getting DataFrame from generator"""
        self.generator.generate_batch(10)
        df = self.generator.get_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertIn('Datetime', df.columns)
        self.assertIn('Close', df.columns)

    def test_continuous_generation(self):
        """Test that generator can continue generating indefinitely"""
        # Generate initial batch
        self.generator.generate_batch(10)
        initial_count = len(self.generator.generated_candles)

        # Generate more
        self.generator.generate_next_candle()
        self.generator.generate_next_candle()

        self.assertEqual(len(self.generator.generated_candles), initial_count + 2)

    def test_timestamp_progression(self):
        """Test that timestamps progress correctly"""
        self.generator.generate_batch(5, interval_minutes=60)
        df = self.generator.get_dataframe()

        # Check that timestamps are sequential
        for i in range(1, len(df)):
            time_diff = (df['Datetime'].iloc[i] - df['Datetime'].iloc[i-1]).total_seconds()
            self.assertEqual(time_diff, 3600)  # 60 minutes in seconds

    def test_volume_is_positive(self):
        """Test that volume is always positive"""
        self.generator.generate_batch(50)
        df = self.generator.get_dataframe()

        self.assertTrue((df['Volume'] >= 0).all())

    def test_price_continuity(self):
        """Test that prices show reasonable continuity"""
        self.generator.generate_batch(100)
        df = self.generator.get_dataframe()

        # Check that close prices don't have extreme jumps
        price_changes = df['Close'].pct_change().dropna()

        # Most changes should be reasonable (< 50%)
        reasonable_changes = (price_changes.abs() < 0.5).sum()
        self.assertGreater(reasonable_changes / len(price_changes), 0.9)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full system"""

    def test_synthetic_data_with_detector(self):
        """Test that synthetic data works with anomaly detector"""
        generator = SyntheticStockGenerator(start_price=100.0)
        detector = AnomalyDetector()

        # Generate data
        data = generator.generate_batch(100)

        # Try to detect anomalies
        for i in range(detector.window_size, len(data)):
            anomaly = detector.detect(data, i)
            if anomaly:
                detector.add_anomaly(anomaly)

        # Should be able to run without errors
        self.assertIsInstance(detector.get_anomalies(), list)

    def test_data_loading_from_csv(self):
        """Test loading real CSV data if available"""
        # This test will skip if no CSV files are available
        data_dir = 'data'
        if not os.path.exists(data_dir):
            self.skipTest("Data directory not found")

        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            self.skipTest("No CSV files found in data directory")

        # Try to load first CSV file
        file_path = os.path.join(data_dir, csv_files[0])
        df = pd.read_csv(file_path)

        # Check basic structure
        self.assertGreater(len(df), 0)
        # Should have OHLCV columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            self.assertIn(col, df.columns)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticStockGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
