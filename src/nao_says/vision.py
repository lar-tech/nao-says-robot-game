from ultralytics import YOLO
import cv2
import numpy as np

class NaoVision:
    def __init__(self, yolo_path="./setup/yolov8n.pt"):
        self.yolo = YOLO(yolo_path)

    def detect_objects(self, image, confidence=0.5, target_objects=None):
        results = self.yolo(image, conf=confidence, verbose=False)
        detected = []
        for result in results:
            if result.boxes is None:
                continue
            for obj in result.boxes:
                class_id = int(obj.cls[0])
                class_name = self.yolo.names[class_id]
                if class_name in target_objects:
                    detected.append({"class": class_name, "confidence": float(obj.conf[0]), "bbox": obj.xyxy[0].tolist()})
        if not detected:
            print("No target objects detected.")
        return detected
