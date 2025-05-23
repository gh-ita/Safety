import redis
import pickle
from pymongo import MongoClient
from datetime import datetime
import base64
import cv2
import numpy as np

# --- Redis Setup ---
r = redis.Redis(host='localhost', port=6379, db=0)

# --- MongoDB Setup ---
client = MongoClient("mongodb+srv://ghitahatimieleve:3rvk3d35tJJl2wsA@cluster0.bmwtzxy.mongodb.net/")
db = client["safety_monitoring"]
collection = db["detections"]

def deserialize_image(frame_bytes):
    # Converts binary (numpy array) to base64 for MongoDB
    _, buffer = cv2.imencode('.jpg', frame_bytes)
    return base64.b64encode(buffer).decode('utf-8')

def dump_from_redis_to_mongo(timestamp, frame, det_cls, det_cnf, det_xywh ):
    try:
        # Convert detections to serializable list of dicts
        detection_list = []
        for i in range(det_cls.shape[0]):
            bbox = det_xywh[i].tolist()
            conf = float(det_cnf[i])
            cls_id = int(det_cls[i])

            detection_list.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": bbox
            })

        document = {
            "timestamp": timestamp,
            "frame": deserialize_image(frame),  # store as base64 string
            "detections": detection_list
        }

        collection.insert_one(document)

    except Exception as e:
        print(f"Error processing payload: {e}")

if __name__ == "__main__":
    dump_from_redis_to_mongo()
