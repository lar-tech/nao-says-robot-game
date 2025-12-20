from ultralytics import YOLO
import cv2
import pytesseract

class NaoVision:
    def __init__(self, yolo_path):
        self.yolo = YOLO(yolo_path)
        self.tesseract_config = r'--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789'

    def detect_objects(self, image, target_objects=None):
        results = self.yolo(image, verbose=False)
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
    
    def detect_number(self, image):
        processed_img = self._preprocess_image(image)
        text = pytesseract.image_to_string(processed_img, config=self.tesseract_config)

        digit = "".join([c for c in text if c.isdigit()])
        return digit
    
    def _preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed_img
    
if __name__ == '__main__':
    vision = NaoVision(model_path="./models/yolov8n.pt")
    img = cv2.imread("images/one.jpg")
    numbers = vision.detect_numbers(img)
    print(numbers)

    detections = vision.detect_objects(img, target_objects=["person", "bottle"])
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText( img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("YOLO Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()