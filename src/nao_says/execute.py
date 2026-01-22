import json
import os
import subprocess
import base64
import io
import re
import cv2
import numpy as np

from PIL import Image
from nao_says.voice import NaoVoiceCommand
from nao_says.vision import NaoVision

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # config
    ROBOT_IP = "192.168.1.102" # 118
    PORT = "9559"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    docker_dir = os.path.abspath(os.path.join(project_root, ".."))
    llm_dir = os.path.join(project_root, "models", "qwen")
    yolo_path = os.path.join(project_root, "models", "yolov8n.pt")
    mnist_path = os.path.join(project_root, "models", "mnist.onnx")

    # init voice recorder
    recorder = NaoVoiceCommand(model_dir=llm_dir)
    robot_command = recorder.record_audio()
    recorder.close()
    print("Robot Command:", robot_command)

    # init vision
    vision = NaoVision(yolo_path=yolo_path, mnist_path=mnist_path)

    # robot_command = json.dumps({"wakeword": True, "action":"capture_frame", "params":{}})

    while True:
        # execute on robot
        cmd = [os.path.join(docker_dir, "run-naoqi.sh"), "python2.7", "src/nao_bundle/execute.py", ROBOT_IP, PORT]
        result = subprocess.run(cmd, input=robot_command, capture_output=True, text=True, check=False)
    #     match = re.search(r'(/9j/[A-Za-z0-9+/]+=*)', result.stdout)
    #     if match:
    #         jpeg_bytes = base64.b64decode(match.group(1))
    #         image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    #         detections = vision.detect_objects(image, confidence=0.5, target_objects=["person", "bottle", "toothbrush", "ball", "chair", "key", "shoe"])
    #         for det in detections:
    #             x1, y1, x2, y2 = map(int, det["bbox"])
    #             label = f'{det["class"]} {det["confidence"]:.2f}'
    #             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(image, label, (x1, max(y1 - 5, 15)), 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
    #         cv2.imshow("Detections", image)
            
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         print("Leaving Loop.")
    #         break
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()