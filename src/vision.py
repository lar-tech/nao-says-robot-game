from ultralytics import YOLO
import cv2
import pytesseract

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
    
    def detect_numbers(self, image, roi=None):
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # crop to ROI
        img = image.copy()
        if roi is not None:
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]

        # preprocess
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # ocr
        config = '--psm 8 -c tessedit_char_whitelist=0123456789'
        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
        results = []
        for i, text in enumerate(data['text']):
            text = text.strip()
            if text and text.isdigit():
                results.append({
                    "number": int(text),
                    "text": text,
                    "confidence": data['conf'][i],
                    "bbox": [
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    ]
                })
        
        return results
    
if __name__ == '__main__':
    vision = NaoVision(model_path="./models/yolov8n.pt")
    img = cv2.imread("images/one.jpg")
    numbers = vision.detect_numbers(img)
    print(numbers)

    # detections = vision.detect_objects(img, target_objects=["person", "bottle"])
    # for det in detections:
    #     x1, y1, x2, y2 = map(int, det["bbox"])
    #     label = f'{det["class"]} {det["confidence"]:.2f}'
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText( img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imshow("YOLO Test", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()