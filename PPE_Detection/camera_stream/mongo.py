from pymongo import MongoClient


client = MongoClient("mongodb+srv://ghitahatimieleve:3rvk3d35tJJl2wsA@cluster0.bmwtzxy.mongodb.net/")
db = client["safety_monitoring"]
collection = db["detections"]

for doc in collection.find().limit(5):  # limit to 5 for quick testing
    print(f"Timestamp: {doc['timestamp']}")
    print(f"Detections: {doc['detections']}")
    print(f"Frame (base64 snippet): {doc['frame'][:100]}...")  # only show part of the base64 string
    print("="*50)