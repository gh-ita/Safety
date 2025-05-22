import redis 
import pickle

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def push_to_queue(timestamp,frame, det_cls, det_cnf, det_xywh ):
    payload = pickle.dumps((timestamp,frame, det_cls, det_cnf, det_xywh))
    r.lpush('detection_queue', payload)

payload = r.lindex('detection_queue', 0)
if payload:
    timestamp, frame, det_cls, det_cnf, det_xywh = pickle.loads(payload)
    print(f"Timestamp: {timestamp}, frame : {frame}, Det_cls: {det_cls}, Det_cnf: {det_cnf}, Det_xywh: {det_xywh}")
else:
    print("No payload in the queue.")
