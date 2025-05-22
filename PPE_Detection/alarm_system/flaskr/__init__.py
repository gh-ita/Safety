import eventlet
eventlet.monkey_patch() 
import os
from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    from .camera_stream import streaming_bp, init_streaming
    app.register_blueprint(streaming_bp)
    
    # Initialize streaming functionality
    socketio = init_streaming(app)
    
    # Store socketio in app config so you can access it later if needed
    app.socketio = socketio
    
    return app