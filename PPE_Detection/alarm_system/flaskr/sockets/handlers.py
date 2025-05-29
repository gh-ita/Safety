from .socketio_setup import socketio

@socketio.on('connect')
def on_connect():
    print("Client connected")

@socketio.on('disconnect')
def on_disconnect():
    print("Client disconnected")

@socketio.on('ping')
def handle_ping(data):
    #print(f"Received ping with: {data}")
    socketio.emit('pong', {'message': 'pong from server'})
