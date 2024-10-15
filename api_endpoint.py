from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from pymongo import MongoClient
from Config import get_env

app = FastAPI()
config = get_env.Settings()

# MongoDB connection setup
connection_string = config.connection_string
client = MongoClient(connection_string)
db = client.analytics  # Use the 'analytics' database
analytic_collection = db.Analytics  # Access the 'Analytics' collection

class BaseData(BaseModel):
    modelId: List[int]  # Expecting list of integers
    currentTime: str
    startTime: str
    endTime: str
    zoneId: str
    sourceId: int
    frameno: int
    fps: int  # Set to 1 FPS
    activities: List[str]
    events: List[str]
    camera_type: str  # Changed to a single string type
    tenantId: str = "LSET"
    duration: float

@app.post("/mongoinsert")
def insert_camera_data(data: BaseData) -> Dict[str, str]:
    data_dict = data.dict()
    try:
        analytic_collection.insert_one(data_dict)  # Insert the data into the collection
        print("Data inserted successfully into MongoDB")
    except Exception as e:
        print(f"Error inserting data: {e}")
        return {"status": "failure", "error": str(e)}
    
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
