import requests
from utils.Re_id_funtions.infer_image import ImageInference
import yaml
import json
import os
import shutil
import cv2
import collections
from datetime import datetime, timedelta
from fastapi import FastAPI
import numpy as np
import pandas as pd
from utils.cv2Operations import cv2_operations
from utils.directoryOperations import directory_operations
from ultralytics import YOLO
from utils.utils import *
import easyocr
from Config import get_env
from pymongo import MongoClient
from utils.timeOperations import time_operations
import time
import os
import paramiko
from Annotation import Detection_Anntation,classification_annnotation
from samples import Detection_sampling,classification_sampling
from wifi import get_public_ip,allow_mongo_access_from_ip
from New_cam import check_and_insert_mac_address
class ApiClass:

    def __init__(self, modelInfo) -> None:
        self.modelbird = YOLO(modelInfo["checkpoints"]["bird_weights"])
        self.waterclass = YOLO(modelInfo["checkpoints"]["water_weights"])
        self.present = {}
        self.cameras = {}
        self.last_save_time = 0
        self.config = get_env.Settings()
        self.weights = modelInfo['weights']['weights_path']
        self.configs = modelInfo['weights']['config_yaml']
        self.inferer = ImageInference(self.weights, self.configs)

        self.activity_dict = {}

        for entry in modelInfo["conditions"]:
            activity = entry["activity"]
            if "items" in entry:
                self.activity_dict[activity] = entry["items"]
            else:
                self.activity_dict[activity] = [entry["item"]]

        connection_string = self.config.connection_string
        client = MongoClient(connection_string)
        database = client["analytics"]
        collection = database["model"]
        model_data_frame = pd.DataFrame(list(collection.find()))

        self.reidentification_names = model_data_frame.loc[
            model_data_frame["modelType"] == "ReIdentification", "name"
        ].tolist()
        self.classification_names = model_data_frame.loc[
            model_data_frame["modelType"] == "classification", "name"
        ].tolist()

    def test_reid(self, file, model_name):
        result = self.inferer.infer(file)
        os.remove(file)
        return result

    def water_classification(self, file):
        results = self.waterclass(source=file)  # predict on an image
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        os.remove(file)
        return names_dict[np.argmax(probs)]
    
    def get_bird_detection(self, file, nextmodel, sourceId, event):
        global tracker_list
        bird_images = {}
        bird_count = 0

        detection_output = self.modelbird.track(source=file, conf=0.25, save=False)[0]
        
        detections = detection_output  # Get the first (and only) prediction output
    
        if detection_output and detection_output.boxes is not None and len(detection_output.boxes) > 0:
            # Proceed with extracting detection information
            class_indices = detection_output.boxes.cls.cpu().numpy().astype(int)
            confidences = detection_output.boxes.conf.cpu().numpy()
            boxes = detection_output.boxes.xyxy.cpu().numpy()
            det_bbox = detection_output[0].boxes.xyxy.cpu().numpy()
            # Check if the model provides tracking IDs
            if hasattr(detection_output.boxes, 'id') and detection_output.boxes.id is not None:
                track_id = detection_output.boxes.id.cpu().numpy()
            else:
                print("No tracking IDs available for the current detection.")
                track_id = None
    
            class_names = [self.modelbird.names[idx] for idx in class_indices]

            # Get the current timestamp and date for organizing files
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            today_date = time.strftime("%Y%m%d")

            # Create the output folder structure (images, labels)
            output_folder = f"./{today_date}"
            images_folder = os.path.join(output_folder, "images")
            labels_folder = os.path.join(output_folder, "labels")
            folder_name = "Bird_images"
            os.makedirs(folder_name, exist_ok=True)
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)
            classes_file = os.path.join(output_folder, "classes.txt")
            if not os.path.exists(classes_file):
                with open(classes_file, "w") as f:
                    for class_name in self.modelbird.names.values():
                        f.write(f"{class_name}\n")

            # Check if 1 minute has passed since the last save
            current_time = time.time()
            save_image = False
            if current_time - self.last_save_time >= 60:  # 60 seconds = 1 minute
                save_image = True
                self.last_save_time = current_time

            # Load the image
            frame = cv2.imread(file)
            if frame is None:
                print(f"Error: Could not read image {file}")
                return

            # Save the captured frame as an image only if a minute has passed
            if save_image:
                image_filename = os.path.join(images_folder, f"{timestamp}.jpg")
                print("Saving image:", image_filename)
                cv2.imwrite(image_filename, frame)

            # Extract bounding box data and annotations
            boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding boxes
            height, width, _ = frame.shape
            annotations = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                class_id = class_indices[i]
                annotations.append(f"{class_id} {(x1 + x2) / (2 * width)} {(y1 + y2) / (2 * height)} {(x2 - x1) / width} {(y2 - y1) / height}")
            
            # Save the annotations only if a minute has passed
            if save_image:
                annotations_filename = os.path.join(labels_folder, f"{timestamp}.txt")
                with open(annotations_filename, "w") as f:
                    f.write("\n".join(annotations) + "\n")

            # Log detected objects
            print(f"Detected classes in {file}: {class_names}")
            dir_ops = directory_operations()

            # Handle additional logic for bird and water detection using dir_ops for cropped images
            things_present = []

            for i, class_name in enumerate(class_names):
                saved_image_path = os.path.join(images_folder, f"{i}_{os.path.basename(file)}")
                if class_name == "waterplate" and nextmodel:
                    relevant_models = set(nextmodel).intersection(self.classification_names)
                    if relevant_models:
                        dir_ops.crop_and_save_image(file, boxes[i], saved_image_path)  # Use dir_ops for cropped image
                        water_result = self.water_classification(saved_image_path)
                        things_present.append(water_result)

                elif class_name == "bird" and nextmodel:
                    if track_id is None:
                        continue
                    
                    f = float(track_id[i])
                    tracker_list.append(f)
                    print("tracker list maintained in the birds", tracker_list)
                    bird_count += 1
                    x, y, x_max, y_max = map(int, det_bbox[0])
                    x, y = max(0, x), max(0, y)
                    x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)
                    w, h = x_max - x, y_max - y
                
                    if w > 0 and h > 0:
                        bird_image = frame[y:y + h, x:x + w]
                        bird_images[f] = bird_image

                    bire = os.path.join(folder_name, f"{timestamp}.jpg")
                    cv2.imwrite(bire, frame)

                    relevant_models = set(nextmodel).intersection(self.reidentification_names)
                    if relevant_models:
                        dir_ops.crop_and_save_image(file, boxes[i], saved_image_path)  # Use dir_ops for cropped image
                        birdname, pred_score = self.test_reid(file=saved_image_path, model_name=relevant_models.pop())
                        print(birdname, pred_score)
                        things_present.append(birdname)
                        class_names[i] = birdname
                        things_present.append(f"{birdname} detected")
                        if f not in track_id_class_map:
                            track_id_prob_map[f] = pred_score
                            track_id_class_map[f] = birdname

            if bird_count == 1:
                track_id = tracker_list[0]
                if track_id in track_id_class_map:
                    sample_data_path = os.path.join(output_directory, track_id_class_map[track_id])
                    print("============================111111111111111111=========================================================", track_id_class_map[track_id])
                    if not os.path.exists(sample_data_path):
                        os.makedirs(sample_data_path)
                    cv2.imwrite(f"{sample_data_path}/{frame_count}_{track_id}_{track_id_class_map[track_id]}.jpg", bird_images[track_id])
            elif bird_count == 2:
                print(track_id_prob_map[tracker_list[1]], track_id_prob_map[tracker_list[0]])
                if track_id_class_map[tracker_list[0]] == track_id_class_map[tracker_list[1]]:
                    if track_id_prob_map[tracker_list[0]] > track_id_prob_map[tracker_list[1]]:
                        track_id_prob_map[tracker_list[1]] = "Wrong"
                        print("done..........")
                        del track_id_class_map[tracker_list[1]]
                    elif track_id_prob_map[tracker_list[1]] > track_id_prob_map[tracker_list[0]]:
                        track_id_prob_map[tracker_list[0]] = "Wrong"
                        del track_id_class_map[tracker_list[0]]
                        print("yes.................")
                for track_id in list(track_id_class_map.keys()):  # Use list() to avoid runtime error
                    if track_id in bird_images:
                        sample_data_path = os.path.join(output_directory, track_id_class_map[track_id])
                        if not os.path.exists(sample_data_path):
                            os.makedirs(sample_data_path)
                            
                        print("=============================222222222222222222========================================================", track_id, track_id_class_map[track_id])
                        cv2.imwrite(f"{sample_data_path}/{frame_count}_{track_id}_{track_id_class_map[track_id]}.jpg", bird_images[track_id])
                    else:
                        print(f"Track ID {track_id} not found in bird_images.") 

            bird_count = 0
            tracker_list = []
            print(things_present)
            return things_present

def reconnect_rtsp(rtsp_url, retry_delay=5):
    """Attempt to reconnect to the RTSP stream."""
    print("Attempting to reconnect to the RTSP stream...")
    cap = None
    while cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Failed to connect. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Reconnected to the RTSP stream.")
    return cap


def progress(filename, size, sent):
    """Callback to print progress of the file being transferred."""
    percent_complete = (sent / size) * 100
    print(f"Transferring {filename}: {percent_complete:.2f}%")


if __name__ == "__main__":
    # Get the public IP address
    public_ip = get_public_ip()

    if 'Error' not in public_ip:
        print(f"My Public IP Address is: {public_ip}")
        # Allow access to MongoDB from this IP
        allow_mongo_access_from_ip(public_ip)
    else:
        print(f"Failed to fetch public IP: {public_ip}")
    config = get_env.Settings()
    rtsp_url = config.rtsp_url
    sourceId = config.source_id
    connection_string = config.connection_string
    client = MongoClient(connection_string)
    db = client.analytics  # Use the 'analytics' database
    source_collection = db.source  # Access the 'source' collection
    modelsCollection = db.model
    analytic_collection = db.Analytics
    print(client)
    print(db.name)
    print(source_collection.name)
    db_name=db.name
    collection_name=source_collection.name
    sourceId=check_and_insert_mac_address(connection_string, db_name, collection_name)
    print("yessssssssssssss,",sourceId)
    url = config.end_p
    url1 = config.mongo
    track_id_prob_map = {}
    track_id_class_map = {}
    frame_count=0
    LABEL_STUDIO_URL = 'https://lset-labelstudio.eizen.ai'
    API_KEY = '78ee35f891b2c71089ef1e07b60ade5cdfcf5bee'


    global tracker_list
    tracker_list=[]
    headers = {
        "Content-Type": "application/json"
    }
    global output_directory
    output_directory = './output' 
    # Read the configuration file
    with open("./Config/apiconfig.yaml", "r") as file:
        modelInfo = yaml.safe_load(file)

    # Instantiate the ApiClass
    api_instance = ApiClass(modelInfo)

    # Query the document with the given _id
    document = source_collection.find_one({"_id": sourceId})
    doc1 = document.get("models")
    modelName = []
    eventslist = []
    activitieslist = []
    
    for i in doc1:
        model_data = modelsCollection.find_one({"_id": i})
        if model_data:
            model_name = model_data.get("name")
            if model_name:
                modelName.append(model_name)
            events = model_data.get("events")
            if events:
                eventslist.extend(events)
            activities = model_data.get("activities")
            if activities:
                activitieslist.extend(activities)

    requiredmodels = {}
    for model in modelInfo["modeltree"]:
        if model["modelname"] in modelName:
            if model["requiredmodel"] not in requiredmodels:
                requiredmodels[model["requiredmodel"]] = [model["modelname"]]
            else:
                requiredmodels[model["requiredmodel"]].append(model["modelname"])

    nextmodel = requiredmodels.get(modelName[0], [])

    # Try to open the RTSP stream
    cap = reconnect_rtsp(rtsp_url)
    
    # Get the frames per second (FPS) of the original video stream
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(original_fps)  # Skip enough frames to achieve 1 FPS
    frame_number = 0

    while True:
        now = datetime.now()
        current_time = now.time()
        current_time1 = datetime.now().strftime("%H:%M")

        # Define the start and end times
        start_time = datetime.strptime("06:00:00", "%H:%M:%S").time()
        end_time = datetime.strptime("18:00:00", "%H:%M:%S").time()
        if current_time1 == "18:37":

            today = datetime.now().strftime('%Y%m%d')
            main_dir = f'./{today}'
            base_dir = './output'
            sampling_ratio = 0.2

            Detection_sampling(main_dir, sampling_ratio)
            classification_sampling(base_dir, sampling_ratio)
            print("!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@"*100)

            PROJECT_TITLE = "Detection"
            Detection_Anntation(LABEL_STUDIO_URL, API_KEY, main_dir, PROJECT_TITLE)

            # For folder-based class names
            PROJECT_TITLE = "classification_data"
            BASE_PATH = "./output"  # Path to the folder with class-based subfolders
            classification_annnotation(LABEL_STUDIO_URL, API_KEY, PROJECT_TITLE, base_dir)
            folders=[base_dir,main_dir,"Bird_images"]
            for folder in folders:
                try:
                    shutil.rmtree(folder)  # Deletes the folder and all its contents
                    print(f"Deleted: {folder}")
                except FileNotFoundError:
                    print(f"{folder} not found")
                except Exception as e:
                    print(f"Error deleting {folder}: {e}")

            break  # Exit the loop after the transfer

        # Check if the current time is within the specified range
        if start_time <= current_time <= end_time:
            ret, frame = cap.read()
            
            # Reconnect if unable to read the frame
            if not ret:
                print("Lost connection to the RTSP stream. Trying to reconnect...")
                cap = reconnect_rtsp(rtsp_url)
                continue

            # Process the frame
            current_time_str = time.strftime("%Y%m%d-%H%M%S")
            file_path = f"temp_frame_{current_time_str}.jpg"
            cv2.imwrite(file_path, frame)
            frame_count+=1
            started= time_operations().get_current_time()

            # Call the get_bird_detection method
            detected_items = api_instance.get_bird_detection(file_path, nextmodel, sourceId, eventslist)
            os.remove(file_path)

            # Handle detected items and perform analytics here (similar to your original processing)
            ended = time_operations().get_current_time()
            time_diff = (ended - started).total_seconds()
            print("ddddddddddddddddddddddddd",time_diff)
            if detected_items is None:
                detected_items = []
            
            print("Detected items:", detected_items)

            # Process detected items and activities
            activities = []
            for condition in modelInfo["conditions"]:
                if condition["type"] == "presence" and condition["item"] in detected_items:
                    activities.append(condition["activity"])
                elif condition["type"] == "list":
                    for item in condition["items"]:
                        if item in detected_items:
                            activities.append(condition["activity"])
                            break

            activities_ = [activity for activity in activities if activity in activitieslist]
            events_ = [activity for activity in eventslist if activity in activities_]
            rawAnalytics = {
                "modelId": doc1,
                "currentTime": str(datetime.now()),  # ISO formatted string for time
                "startTime": str(datetime.now()),
                "endTime": str(datetime.now()),
                "zoneId": document['zoneId'],  # String value
                "sourceId": sourceId,  # Integer
                "frameno": frame_number,  # Integer
                "fps": 1,  # Integer, set to 1 FPS
                "activities": activities_,  # List of strings
                "events": events_,  # List of strings
                "camera_type": document['sourceType'],  # Single string
                "tenantId": "LSET",
                "duration": time_diff
            }

            response1 = requests.post(url1, json=rawAnalytics)
            print(rawAnalytics)
            print("====================================================================")
            data = {
                "activities": activities_,
                "duration": time_diff,
                "currentTime": str(datetime.now()),
                "endTime": str(datetime.now()),
                "startTime": str(datetime.now()),
                "events": events_
            }

            response = requests.post(url, headers=headers, json=data)
            print(f"Status Code: {response.status_code}")

            # Skip the required number of frames to achieve 1 FPS
            for _ in range(frame_skip - 1):
                cap.grab()  # Grabbing the frame without processing
            time.sleep(1)
            print("only one frame.....................................")
        else:
            print("Outside of allowed operational hours. Waiting for the next allowed time...")
            time.sleep(60)  # Wait for 1 minute before checking again

    cap.release()
    cv2.destroyAllWindows()