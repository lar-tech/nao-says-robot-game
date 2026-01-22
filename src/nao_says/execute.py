import os
import subprocess
import base64
import re
import cv2
import numpy as np
import urllib.request

from nao_says.voice import NaoVoiceCommand
from nao_says.vision import NaoVision

# config
ROBOT_IP = "192.168.1.102" # 192.168.1.102 or 192.168.1.118
PORT = "9559"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
docker_dir = os.path.abspath(os.path.join(project_root, ".."))
yolo_path = "./setup/yolov8n.pt"
if not os.path.exists(yolo_path):
    print("Downloading YOLO model...")
    urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt", yolo_path)
audio_dir = "./setup/whisper"

# voice recorder and vision
recorder = NaoVoiceCommand(model_dir=audio_dir)
vision = NaoVision(yolo_path=yolo_path)

while True:
    command = recorder.get_command()
    print(f"Executing: {command}")
    
    # execute on robot
    cmd = [os.path.join(docker_dir, "run-naoqi.sh"), "python2.7", "src/nao_bundle/execute.py", ROBOT_IP, PORT]
    result = subprocess.run(cmd, input=command, capture_output=True, text=True, check=False)
    print(result.stdout)
    
    # vision processing
    match = re.search(r'(/9j/[A-Za-z0-9+/]+=*)', result.stdout)
    if match:
        jpeg_bytes = base64.b64decode(match.group(1))
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        detections = vision.detect_objects(image, confidence=0.5, target_objects=["person", "bottle", "ball", "chair", "key", "shoe"])
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f'{det["class"]} {det["confidence"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(y1 - 5, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Detections", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Leaving Loop.")
            cv2.destroyAllWindows()
            break