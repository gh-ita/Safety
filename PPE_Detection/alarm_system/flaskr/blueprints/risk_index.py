from flask import request, jsonify, Blueprint, current_app
from PPE_Detection.alarm_system.flaskr.redis_jobs import jobs


risk_idx_bp = Blueprint('risk_idx', __name__)

@risk_idx_bp.route('/update-scheduler-job', methods=['POST'])
def update_scheduled_job():
    data = request.get_json()
    print("Received updated class_risk:", data)
    risk_idx= data.get('risk_idx')

    # Remove existing job if it exists
    if current_app.scheduler.get_job('my_job'):
        current_app.scheduler.remove_job('my_job')

    # Reschedule with new args
    current_app.scheduler.add_job(
        id='process_detection_queue',
        func= jobs.process_detection_queue,
        trigger='interval',
        seconds=5,
        args=[risk_idx, current_app.socketio],
    )

    return jsonify({"status": "Job updated with class risks"}), 200
