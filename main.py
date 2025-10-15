"""
Flask Application with WebSocket for Real-time Stock Data Streaming
No page reloads, no scrolling issues!
"""
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import json
import time
import threading
import os

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
    'data': None
}

def load_stock_data(file_path):
    """Load stock data from CSV"""
    df = pd.read_csv(file_path)
    if 'Datetime' not in df.columns and 'Date' in df.columns:
        df = df.rename(columns={'Date': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    return df

def stream_data():
    """Background thread that streams data via WebSocket"""
    while True:
        if streaming_state['active'] and not streaming_state['paused'] and streaming_state['data'] is not None:
            data = streaming_state['data']

            if streaming_state['current_index'] < len(data):
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
                streaming_state['current_index'] = min(
                    streaming_state['current_index'] + increment,
                    len(data)
                )

                # Get windowed data
                window_size = streaming_state['window_size']
                start_idx = max(0, streaming_state['current_index'] - window_size)
                window_data = data.iloc[start_idx:streaming_state['current_index']]

                # Prepare data for frontend
                chart_data = {
                    'datetime': window_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'open': window_data['Open'].tolist(),
                    'high': window_data['High'].tolist(),
                    'low': window_data['Low'].tolist(),
                    'close': window_data['Close'].tolist(),
                    'volume': window_data['Volume'].tolist(),
                    'current_index': streaming_state['current_index'],
                    'total_points': len(data),
                    'current_price': float(data['Close'].iloc[streaming_state['current_index'] - 1]),
                    'start_price': float(data['Close'].iloc[0])
                }

                # Emit data update
                socketio.emit('data_update', chart_data)

                # Calculate delay
                delay = 1.0 / streaming_state['speed_multiplier']
                time.sleep(delay)
            else:
                # Reached end
                streaming_state['active'] = False
                socketio.emit('streaming_complete')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    """Serve the main page"""
    # Get available data files
    data_dir = 'data'
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    return render_template('index.html', data_files=data_files)

@app.route('/api/load_file', methods=['POST'])
def load_file():
    """Load a data file"""
    filename = request.json.get('filename')
    file_path = os.path.join('data', filename)

    try:
        data = load_stock_data(file_path)
        streaming_state['data'] = data
        streaming_state['selected_file'] = filename
        streaming_state['current_index'] = 0

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

if __name__ == '__main__':
    # Start background streaming thread
    thread = threading.Thread(target=stream_data, daemon=True)
    thread.start()

    # Run Flask app
    print("🚀 Starting Flask app on http://localhost:8080")
    print("📊 Stock streaming with NO page reloads!")
    socketio.run(app, debug=True, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)
