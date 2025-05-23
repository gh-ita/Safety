import eventlet
eventlet.monkey_patch() 
import os
from flask import Flask
from flask_rq2 import RQ
from sockets.socketio_setup import socketio
from apscheduler.schedulers.background import BackgroundScheduler
from redis_jobs.jobs import process_detection_queue

rq = RQ()
scheduler = BackgroundScheduler()

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config['REDIS_URL'] = 'redis://localhost:6379/0'
    rq.init_app(app)
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

    def start_scheduler():
        if not scheduler.running:
            scheduler.add_job(process_detection_queue.queue, 'interval', seconds=10, id='process_detection_queue')
            scheduler.start()

    # Optionally shut down scheduler with app
    @app.teardown_appcontext
    def shutdown_scheduler(exception=None):
        if scheduler.running:
            scheduler.shutdown()

    from .blueprints.camera_stream import streaming_bp, init_streaming
    app.register_blueprint(streaming_bp)
    

    streaming = init_streaming(app)
    socketio = socketio.init_app(app)

    # Store socketio in app config so you can access it later if needed
    app.socketio = socketio
    start_scheduler()

    return app