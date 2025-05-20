from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@app.route('/')
def index():
    return render_template('index.html')

# Function to encode and emit frames
def send_frame(frame):
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # Convert to base64 for sending over socketio
    frame_bytes = base64.b64encode(buffer).decode('utf-8')
    # Send to all connected clients
    socketio.emit('video_frame', {'frame': frame_bytes})

# Thread to capture and process frames
def video_feed():
    while True:
        # Your ZED camera processing code here
        # Replace cv2.imshow with:
        send_frame(global_image)
        time.sleep(0.03)  # ~30 FPS

if __name__ == '__main__':
    # Start camera thread
    thread = threading.Thread(target=video_feed)
    thread.daemon = True
    thread.start()
    
    # Start Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000)