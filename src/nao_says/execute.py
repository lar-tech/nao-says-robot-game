import os
import subprocess
import base64
import re
import cv2
import numpy as np
import urllib.request
import json

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
    command_json = recorder.get_command()
    command = json.loads(command_json)
    print(f"Executing: {command}")

    # execute on robot
    cmd = [os.path.join(docker_dir, "run-naoqi.sh"), "python2.7", "src/nao_bundle/execute.py", ROBOT_IP, PORT]
    result = subprocess.run(cmd, input=command_json, capture_output=True, text=True, check=False)
    
    # vision processing
    match = re.search(r'(/9j/[A-Za-z0-9+/]+=*)', result.stdout)
    if match:
        jpeg_bytes = base64.b64decode(match.group(1))
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        detections = vision.detect_objects(image, confidence=0.5, target_objects=["person", "bottle", "ball", "chair", "key", "shoe", "cup"])

        # count detected objects by class
        if detections:
            counts = {}
            for det in detections:
                cls = det["class"]
                counts[cls] = counts.get(cls, 0) + 1

            # build description text
            parts = []
            for cls, count in counts.items():
                if count == 1:
                    parts.append(f"1 {cls}")
                else:
                    parts.append(f"{count} {cls}s")

            description = "I see " + ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else parts[0] if parts else "")

            # send say_text command to robot
            say_command = json.dumps({'wakeword': True, 'action': 'say_text', 'params': {'text': description}})
            cmd = [os.path.join(docker_dir, "run-naoqi.sh"), "python2.7", "src/nao_bundle/execute.py", ROBOT_IP, PORT]
            subprocess.run(cmd, input=say_command, capture_output=True, text=True, check=False)

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

    if command["action"] == "game_over":
        print("Game over detected. Exiting loop.")
        break
    command_json = {}