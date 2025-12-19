import base64
import os
import subprocess

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from src.voice import NaoVoiceCommand
from src.vision import NaoVision

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # config
    ROBOT_IP = "192.168.1.118"
    PORT = "9559"
    LLM_DIR = "./models/qwen"
    YOLO_PATH = "./models/yolov8n.pt"
    IMAGE_DIR = "./images"
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    if not os.path.exists(LLM_DIR):
        local_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir=LLM_DIR, local_dir_use_symlinks=False)
        print("Model downloaded to:", local_dir)

    # get voice command
    recorder = NaoVoiceCommand(model_dir=LLM_DIR)
    command = recorder.record_audio()
    recorder.close()

    # execute on robot
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [parent_dir + "/run-naoqi.sh", "python2.7", "src/execute.py", ROBOT_IP, PORT, command]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # decode image
    vision = NaoVision(model_path=YOLO_PATH)
    jpeg_bytes = base64.b64decode(result.stdout.strip())
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    detections = vision.detect_objects(img,target_objects=["person", "bottle"])
    
    # results
    print("returncode:", result.returncode)
    print("stdout:\n", result.stdout)
    print("stderr:\n", result.stderr)

