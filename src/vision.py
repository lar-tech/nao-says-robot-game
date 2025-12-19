from ultralytics import YOLO

class NaoVision:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image, target_objects=None):
        results = self.model(image, verbose=False)
        detected = []
        for result in results:
            if result.boxes is None:
                continue
            for obj in result.boxes:
                class_id = int(obj.cls[0])
                class_name = self.model.names[class_id]
                if target_objects is None or class_name in target_objects:
                    detected.append({"class": class_name, "confidence": float(obj.conf[0]), "bbox": obj.xyxy[0].tolist()})
        return detected
    
if __name__ == '__main__':
    import cv2

    vision = NaoVision(model_path="./models/yolov8n.pt")
    img = cv2.imread("images/bottle.jpg")
    detections = vision.detect_objects(img, target_objects=["person", "bottle"])
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText( img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("YOLO Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()