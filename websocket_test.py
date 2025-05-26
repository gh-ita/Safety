import socketio

# Replace this URL with your actual server URL and port
SOCKETIO_SERVER_URL = 'http://localhost:5000'  # or the appropriate IP and port

sio = socketio.Client()

@sio.event
def connect():
    print("Connected to the WebSocket server.")

@sio.event
def disconnect():
    print("Disconnected from the server.")

@sio.on('high_severity_alert')
def on_high_severity_alert(data):
    print("Received high_severity_alert:")
    print(data)

try:
    sio.connect(SOCKETIO_SERVER_URL)
    sio.wait()  # Keep the client alive to listen for events
except Exception as e:
    print(f"Connection failed: {e}")
