from ultralytics import YOLO
import cv2
import numpy as np

class NaoVision:
    def __init__(self, yolo_path="./setup/yolov8n.pt"):
        self.yolo = YOLO(yolo_path)

    # object detection
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
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    vision = NaoVision(yolo_path="./setup/yolov8n.pt")

    # object detection
    object_image = cv2.imread("images/person.jpg")
    detections = vision.detect_objects(object_image, target_objects=["person", "bottle", "toothbrush"])
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.rectangle(object_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(object_image, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    plt.figure()
    plt.imshow(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'{label}')
    plt.show()
