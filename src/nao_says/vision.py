from ultralytics import YOLO
import cv2
import numpy as np
import onnxruntime as ort

class NaoVision:
    def __init__(self, yolo_path="./models/yolov8n.pt", mnist_path="./models/mnist.onnx"):
        self.yolo = YOLO(yolo_path)
        
        # mnist model
        self.sess = ort.InferenceSession(mnist_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    # object detection
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
    
    # number detection
    @staticmethod
    def _preprocess_image(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted_img = cv2.bitwise_not(bin_img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel, iterations=1)

        return processed_img
    
    @staticmethod
    def _find_contours(processed_img):
        cnts, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        H, W = processed_img.shape[:2]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 150:
                continue
            if h < 15 or w < 5:
                continue
            boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: b[0])

        return boxes

    @staticmethod
    def _scale_image(processed_img, box):
        # extract ROI
        x, y, w, h = box
        roi = processed_img[y:y+h, x:x+w]
        ys, xs = np.where(roi > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        roi = roi[y0:y1+1, x0:x1+1]

        # scale in 20x20 box
        h2, w2 = roi.shape
        if h2 > w2:
            new_h = 20
            new_w = max(1, int(round(w2 * (20.0 / h2))))
        else:
            new_w = 20
            new_h = max(1, int(round(h2 * (20.0 / w2))))
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = roi

        x = canvas.astype(np.float32) / 255.0
        x = x[None, None, :, :]
        return x

    @staticmethod
    def _softmax(v):
        v = v - np.max(v)
        e = np.exp(v)
        return e / np.sum(e)
    
    def _predict_digit(self, scaled_img):
        logits = self.sess.run([self.output_name], {self.input_name: scaled_img})[0]  # (1,10)
        probs = self._softmax(logits[0])
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return pred, conf
    
    def detect_numbers(self, raw_image):
        processed_img = self._preprocess_image(raw_image)
        boxes = self._find_contours(processed_img)

        digits = []
        digit_infos = []
        for box in boxes:
            scaled_img = self._scale_image(processed_img, box)
            if scaled_img is None:
                continue
            pred, conf = self._predict_digit(scaled_img)

            if conf < 0.60:
                pred = "?"
            digits.append(str(pred))
            digit_infos.append({"box": box, "pred": pred, "conf": conf})
        
        numbers = "".join(digits)

        vis = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        for info in digit_infos:
            x, y, w, h = info["box"]
            label = f'{info["pred"]} ({info["conf"]:.2f})'
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, label, (x, max(0, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return numbers, vis

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    vision = NaoVision(yolo_path="./models/yolov8n.pt", mnist_path="./models/mnist.onnx")

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
    
    # number detection
    raw_image = cv2.imread("images/numbers.jpg")
    numbers, vis = vision.detect_numbers(raw_image)
    
    plt.figure()
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()