#!/bin/bash
# Test runner script for Anomaly Detector

echo "=================================="
echo "Running Anomaly Detector Tests"
echo "=================================="
echo ""

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    echo "Activating virtual environment..."
    source myenv/bin/activate
fi

# Run tests
echo "Running test suite..."
python tests/test_anomaly_detector.py

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed. Please review the output above."
fi

exit $EXIT_CODE
