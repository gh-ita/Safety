import eventlet
eventlet.monkey_patch()
import os
from flask import Flask
from flask_rq2 import RQ
from PPE_Detection.alarm_system.flaskr.sockets.socketio_setup import socketio
from apscheduler.schedulers.background import BackgroundScheduler
from PPE_Detection.alarm_system.flaskr.redis_jobs import jobs

rq = RQ()
scheduler = BackgroundScheduler()
rq.job(jobs.process_detection_queue)

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
            scheduler.add_job(jobs.process_detection_queue, 'interval', seconds=5, id='process_detection_queue')
            scheduler.start()


    @app.teardown_appcontext
    def shutdown_scheduler(exception=None):
        if scheduler.running:
            scheduler.shutdown()

    from .blueprints.camera_stream import streaming_bp, init_streaming
    from .blueprints.history import history_bp
    app.register_blueprint(streaming_bp)
    app.register_blueprint(history_bp)

    streaming = init_streaming(app)
    socketio.init_app(app)

    # Store socketio in app config so you can access it later if needed
    app.socketio = socketio
    start_scheduler()

    return app