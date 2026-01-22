import os
import urllib.request

def main():
    # paths and urls
    os.makedirs("./models", exist_ok=True)
    yolo_path = "./models/yolov8n.pt"
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    # yolo
    if not os.path.exists(yolo_path):
        print("Downloading YOLO model...")
        urllib.request.urlretrieve(yolo_url, yolo_path)
    else:
        print("YOLO model already present.")

if __name__ == "__main__":
    main()