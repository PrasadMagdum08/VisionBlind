from ultralytics import YOLO
import numpy as np

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

def detect_objects(image):
    results = model(image)
    names = results[0].names
    detections = results[0].boxes.cls.tolist()
    detected_labels = list(set([names[int(cls)] for cls in detections]))
    return detected_labels
