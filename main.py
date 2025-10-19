"""
Flask Application with WebSocket for Real-time Stock Data Streaming
No page reloads, no scrolling issues!
"""
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import json
import time
import threading
import os
from anomaly import AnomalyDetector, AnomalyType, Anomaly, AnomalySeverity
from persistent_random_data import SyntheticStockGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock_streaming_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
streaming_state = {
    'active': False,
    'paused': False,
    'current_index': 0,
    'speed_multiplier': 1,
    'window_size': 50,
    'selected_file': None,
    'data': None,
    'anomaly_detector': None,
    'is_synthetic': False,  # Flag to indicate synthetic data mode
    'synthetic_generator': None,  # Generator instance for synthetic data
    'enabled_anomaly_types': {
        'price_spike': True,
        'price_drop': True,
        'volume_spike': True,
        'volatility_spike': True,
        'gap': True
    },
    'anomaly_config': {
        'window_size': 20,
        'z_threshold': 3.0,
        'volume_threshold': 2.5,
        'volatility_threshold': 2.0
    },
    'custom_anomalies': []
}

def load_stock_data(file_path):
    """Load stock data from CSV - no gap filling, use raw data only"""
    df = pd.read_csv(file_path)

    # Handle different datetime column names
    if 'Datetime' not in df.columns and 'Date' in df.columns:
        df = df.rename(columns={'Date': 'Datetime'})

    # Convert to datetime and remove timezone info for consistency
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_localize(None)
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Remove duplicate consecutive values (closed hours)
    # Keep only rows where price actually changes
    if len(df) > 1:
        # Calculate if values changed from previous row
        changed = (
            (df['Open'] != df['Open'].shift(1)) |
            (df['High'] != df['High'].shift(1)) |
            (df['Low'] != df['Low'].shift(1)) |
            (df['Close'] != df['Close'].shift(1))
        )
        # Always keep the first row
        changed.iloc[0] = True
        # Filter to only changed rows
        df = df[changed].copy()

    return df

def execute_custom_anomaly(code, window_data, current_row, current_index):
    """
    Safely execute custom anomaly detection code

    Returns:
        tuple: (detected: bool, description: str, severity: int)
    """
    try:
        # Create restricted namespace for execution
        namespace = {
            'pd': pd,
            'np': np,
            'window_data': window_data,
            'current_row': current_row,
            'current_index': current_index,
            'len': len,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'float': float,
            'int': int,
            'str': str,
            'description': '',
            'severity': 2,  # Default to medium
            'detected': False,
            '__builtins__': {
                'True': True,
                'False': False,
                'None': None,
                'range': range,
                'enumerate': enumerate,
            }
        }

        # Wrap code to capture return value
        wrapped_code = f"""
result = None
def _detect():
{chr(10).join('    ' + line for line in code.split(chr(10)))}

try:
    result = _detect()
except:
    pass

if result is not None:
    detected = result
"""

        # Execute the code
        exec(wrapped_code, namespace)

        # Get results
        detected = namespace.get('detected', False)
        description = namespace.get('description', 'Custom anomaly detected')
        severity = namespace.get('severity', 2)

        return detected, description, severity

    except Exception as e:
        print(f"Error executing custom anomaly code: {e}")
        import traceback
        traceback.print_exc()
        return False, '', 0

def stream_data():
    """Background thread that streams data via WebSocket"""
    while True:
        # Check pause/stop state at the very beginning of each iteration
        if not streaming_state['active'] or streaming_state['paused'] or streaming_state['data'] is None:
            time.sleep(0.1)
            continue

        data = streaming_state['data']

        # Calculate increment based on speed
        speed = streaming_state['speed_multiplier']
        if speed >= 100:
            increment = 50
        elif speed >= 10:
            increment = 20
        elif speed >= 5:
            increment = 10
        elif speed >= 3:
            increment = 5
        elif speed >= 2:
            increment = 3
        else:
            increment = 1

        # Increment index
        streaming_state['current_index'] += increment

        # Handle synthetic data generation
        if streaming_state['is_synthetic'] and streaming_state['synthetic_generator'] is not None:
            # Generate new candles as needed (ensure we have enough data)
            generator = streaming_state['synthetic_generator']
            while len(generator.generated_candles) < streaming_state['current_index']:
                generator.generate_next_candle(interval_minutes=60)

            # Update data reference
            data = generator.get_dataframe()
            streaming_state['data'] = data

        # Check if we have enough data
        if streaming_state['current_index'] <= len(data):
            # Get all data from start to current index (so user can scroll back)
            all_data = data.iloc[0:streaming_state['current_index']]
        else:
            # No more data and not synthetic - stop
            if not streaming_state['is_synthetic']:
                streaming_state['active'] = False
                socketio.emit('streaming_complete')
                continue
            else:
                # This shouldn't happen for synthetic, but just in case
                continue

        # Detect anomalies if detector is enabled
        current_anomalies = []
        if streaming_state['anomaly_detector'] is not None:
            detector = streaming_state['anomaly_detector']
            anomaly = detector.detect(data, streaming_state['current_index'] - 1)

            # Check if anomaly type is enabled
            if anomaly:
                enabled = streaming_state['enabled_anomaly_types']
                if enabled.get(anomaly.type.value, True):
                    detector.add_anomaly(anomaly)

            # Check custom anomalies
            window_size = streaming_state['anomaly_config']['window_size']
            window_start = max(0, streaming_state['current_index'] - window_size)
            window_data = data.iloc[window_start:streaming_state['current_index']]
            current_row = data.iloc[streaming_state['current_index'] - 1]

            for custom in streaming_state['custom_anomalies']:
                if not custom.get('enabled', True):
                    continue

                detected, description, severity = execute_custom_anomaly(
                    custom['code'],
                    window_data,
                    current_row,
                    streaming_state['current_index'] - 1
                )

                if detected:
                    # Create custom anomaly object
                    custom_anomaly = Anomaly(
                        index=streaming_state['current_index'] - 1,
                        timestamp=current_row['Datetime'],
                        type=AnomalyType.PRICE_SPIKE,  # Use generic type for custom
                        severity=AnomalySeverity(min(max(severity, 1), 4)),
                        value=float(current_row['Close']),
                        baseline=float(window_data['Close'].mean()),
                        deviation=0.0,
                        description=f"[{custom['name']}] {description}",
                        metrics={'custom': True, 'name': custom['name']}
                    )
                    detector.add_anomaly(custom_anomaly)

            # Get all anomalies detected so far
            all_anomalies = detector.get_anomalies()
            for a in all_anomalies:
                if a.index < streaming_state['current_index']:
                    current_anomalies.append(a.to_dict())

        # Prepare data for frontend (send all data streamed so far)
        chart_data = {
            'datetime': all_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': all_data['Open'].tolist(),
            'high': all_data['High'].tolist(),
            'low': all_data['Low'].tolist(),
            'close': all_data['Close'].tolist(),
            'volume': all_data['Volume'].tolist(),
            'current_index': streaming_state['current_index'],
            'total_points': len(data),
            'current_price': float(data['Close'].iloc[streaming_state['current_index'] - 1]),
            'start_price': float(data['Close'].iloc[0]),
            'anomalies': current_anomalies,
            'anomaly_count': len(detector.get_anomalies()) if streaming_state['anomaly_detector'] else 0
        }

        # Emit data update
        socketio.emit('data_update', chart_data)

        # Calculate delay
        delay = 1.0 / streaming_state['speed_multiplier']
        time.sleep(delay)

@app.route('/')
def index():
    """Serve the main page"""
    # Get available data files
    data_dir = 'data'
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    # Add synthetic option at the beginning
    data_files.insert(0, 'SYNTHETIC_continuous')

    return render_template('index.html', data_files=data_files)

@app.route('/api/load_file', methods=['POST'])
def load_file():
    """Load a data file or initialize synthetic data generator"""
    filename = request.json.get('filename')

    try:
        # Check if synthetic mode
        if filename == 'SYNTHETIC_continuous':
            # Find a reference file to learn patterns from
            data_dir = 'data'
            reference_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            reference_file = None
            if reference_files:
                reference_file = os.path.join(data_dir, reference_files[0])

            # Initialize synthetic generator
            generator = SyntheticStockGenerator(reference_file=reference_file)

            # Generate initial batch of data
            generator.generate_batch(num_candles=100, interval_minutes=60)

            streaming_state['is_synthetic'] = True
            streaming_state['synthetic_generator'] = generator
            streaming_state['data'] = generator.get_dataframe()
            streaming_state['selected_file'] = filename
            streaming_state['current_index'] = 0

            data = streaming_state['data']

            # Initialize anomaly detector
            config = streaming_state['anomaly_config']
            streaming_state['anomaly_detector'] = AnomalyDetector(
                window_size=config['window_size'],
                z_threshold=config['z_threshold'],
                volume_threshold=config['volume_threshold'],
                volatility_threshold=config['volatility_threshold']
            )

            return jsonify({
                'success': True,
                'total_points': 'Infinite (Synthetic)',
                'date_from': data['Datetime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                'date_to': 'Continuous generation...'
            })
        else:
            # Load regular CSV file
            file_path = os.path.join('data', filename)
            data = load_stock_data(file_path)

            streaming_state['is_synthetic'] = False
            streaming_state['synthetic_generator'] = None
            streaming_state['data'] = data
            streaming_state['selected_file'] = filename
            streaming_state['current_index'] = 0

            # Initialize anomaly detector with current config
            config = streaming_state['anomaly_config']
            streaming_state['anomaly_detector'] = AnomalyDetector(
                window_size=config['window_size'],
                z_threshold=config['z_threshold'],
                volume_threshold=config['volume_threshold'],
                volatility_threshold=config['volatility_threshold']
            )

            return jsonify({
                'success': True,
                'total_points': len(data),
                'date_from': data['Datetime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                'date_to': data['Datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('start_streaming')
def handle_start():
    """Start streaming"""
    streaming_state['active'] = True
    streaming_state['paused'] = False
    emit('status_update', {'status': 'streaming'})

@socketio.on('pause_streaming')
def handle_pause():
    """Pause streaming"""
    streaming_state['paused'] = not streaming_state['paused']
    status = 'paused' if streaming_state['paused'] else 'streaming'
    emit('status_update', {'status': status})

@socketio.on('stop_streaming')
def handle_stop():
    """Stop streaming"""
    streaming_state['active'] = False
    streaming_state['paused'] = False
    streaming_state['current_index'] = 0
    emit('status_update', {'status': 'stopped'})

@socketio.on('reset_streaming')
def handle_reset():
    """Reset streaming"""
    streaming_state['current_index'] = 0
    streaming_state['active'] = False
    streaming_state['paused'] = False
    emit('status_update', {'status': 'reset'})

@socketio.on('set_speed')
def handle_speed(data):
    """Set streaming speed"""
    streaming_state['speed_multiplier'] = data['speed']

@socketio.on('set_window_size')
def handle_window_size(data):
    """Set window size"""
    streaming_state['window_size'] = data['size']

@socketio.on('toggle_anomaly_type')
def handle_toggle_anomaly_type(data):
    """Toggle anomaly detection type on/off"""
    anomaly_type = data['type']
    enabled = data['enabled']
    streaming_state['enabled_anomaly_types'][anomaly_type] = enabled
    emit('anomaly_type_toggled', {'type': anomaly_type, 'enabled': enabled})

@socketio.on('update_anomaly_config')
def handle_anomaly_config(data):
    """Update anomaly detection configuration"""
    config = streaming_state['anomaly_config']

    if 'window_size' in data:
        config['window_size'] = data['window_size']
    if 'z_threshold' in data:
        config['z_threshold'] = data['z_threshold']
    if 'volume_threshold' in data:
        config['volume_threshold'] = data['volume_threshold']
    if 'volatility_threshold' in data:
        config['volatility_threshold'] = data['volatility_threshold']

    # Recreate detector with new config
    if streaming_state['anomaly_detector'] is not None:
        streaming_state['anomaly_detector'] = AnomalyDetector(
            window_size=config['window_size'],
            z_threshold=config['z_threshold'],
            volume_threshold=config['volume_threshold'],
            volatility_threshold=config['volatility_threshold']
        )

    emit('config_updated', config)

@socketio.on('clear_anomalies')
def handle_clear_anomalies():
    """Clear all detected anomalies"""
    if streaming_state['anomaly_detector'] is not None:
        streaming_state['anomaly_detector'].clear_anomalies()
    emit('anomalies_cleared')

@socketio.on('add_custom_anomaly')
def handle_add_custom_anomaly(data):
    """Add a custom anomaly detector"""
    streaming_state['custom_anomalies'].append(data)
    emit('custom_anomaly_added', {'success': True, 'id': data['id']})

@socketio.on('delete_custom_anomaly')
def handle_delete_custom_anomaly(data):
    """Delete a custom anomaly detector"""
    anomaly_id = data['id']
    streaming_state['custom_anomalies'] = [
        a for a in streaming_state['custom_anomalies'] if a['id'] != anomaly_id
    ]
    emit('custom_anomaly_deleted', {'success': True, 'id': anomaly_id})

@socketio.on('toggle_custom_anomaly')
def handle_toggle_custom_anomaly(data):
    """Toggle a custom anomaly detector on/off"""
    anomaly_id = data['id']
    enabled = data['enabled']
    for anomaly in streaming_state['custom_anomalies']:
        if anomaly['id'] == anomaly_id:
            anomaly['enabled'] = enabled
            break
    emit('custom_anomaly_toggled', {'success': True, 'id': anomaly_id, 'enabled': enabled})

if __name__ == '__main__':
    # Start background streaming thread
    thread = threading.Thread(target=stream_data, daemon=True)
    thread.start()

    # Get port from environment variable (for deployment) or default to 8080
    port = int(os.environ.get('PORT', 8080))

    # Run Flask app
    print(f"🚀 Starting Flask app on port {port}")
    print("📊 Stock streaming with NO page reloads!")
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
