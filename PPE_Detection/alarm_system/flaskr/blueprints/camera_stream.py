from flask import Blueprint, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time
import PPE_Detection.shared_state as shared_state
from PPE_Detection.alarm_system.flaskr.sockets.socketio_setup import socketio


streaming_bp = Blueprint('streaming', __name__)

@streaming_bp.route('/streaming')
def stream_page():
    return render_template('stream.html')  # Renamed to avoid conflicts

# Function to encode and emit frames
def send_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_bytes = base64.b64encode(buffer).decode('utf-8')
    print("Emitting frame...") 
    socketio.emit('video_frame', {'frame': frame_bytes})

# Thread to capture and process frames
def video_feed():
    while True:
        print("Capturing frame...")
        with shared_state.lock:
            frame = shared_state.global_image
            if frame is not None:
                send_frame(frame.copy())
                print("Frame sent")
            if frame is None:
                print("No image to send")
        time.sleep(0.03)

def init_streaming(app):
    """Initialize streaming functionality with the app"""
    # Start camera thread
    thread = threading.Thread(target=video_feed)
    thread.daemon = True
    thread.start()