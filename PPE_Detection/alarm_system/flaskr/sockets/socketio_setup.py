from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*",async_mode='eventlet', logger=True, engineio_logger=True,message_queue='redis://localhost:6379/')
