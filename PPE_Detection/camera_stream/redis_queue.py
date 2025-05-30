import redis 
import pickle

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)
#

def push_to_queue(timestamp,frame, det_cls, det_cnf, det_xywh ):
    payload = pickle.dumps((timestamp,frame, det_cls, det_cnf, det_xywh))
    r.lpush('detection_queue', payload)



if __name__ == "__main__":
    N = 10
    total_items = r.llen('history_queue')
    start = max(total_items - N, 0)
    print(start)
    end = total_items - 1
    print(end)
    items = r.lrange('history_queue', 0, 20)  # Oldest to newest (recent N)
    
    # Reverse to show newest first
    for i, item in enumerate(reversed(items), 1):
        try:
            data = pickle.loads(item)
            timestamp = data[0]
            severity = data[2]
            print(f"Item {i}: Timestamp = {timestamp}, Severity = {severity}")
        except Exception as e:
            print(f"Item {i}: Failed to decode - {e}")





