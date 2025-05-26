from flask import Blueprint, jsonify
import pickle
from PPE_Detection.camera_stream.redis_queue import r

history_bp = Blueprint('history', __name__)

@history_bp.route('/api/history', methods=['GET'])
def get_history():
    N = 10  
    history = []
    
    items = r.lrange('history_queue', 0, 10)
    print(f"[DEBUG] Fetched {len(items)} items from history_queue")

    for i, item in enumerate(reversed(items)):
        try:
            timestamp, frame_bytes, severity_index, class_counts = pickle.loads(item)
            history.append({
                "timestamp": timestamp,
                "severity": severity_index,
                "class_counts": class_counts,
                "frame": frame_bytes 
            })
        except Exception as e:
            print(f"[ERROR] Failed to load item {i} from history_queue: {e}")
            continue

    return jsonify(history)
