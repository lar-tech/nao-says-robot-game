import json
import os
import subprocess

from nao_says.voice import NaoVoiceCommand
from nao_says.vision import NaoVision

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # config
    ROBOT_IP = "192.168.1.118"
    PORT = "9559"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    docker_dir = os.path.abspath(os.path.join(project_root, ".."))
    llm_dir = os.path.join(project_root, "models", "qwen")
    yolo_path = os.path.join(project_root, "models", "yolov8n.pt")
    mnist_path = os.path.join(project_root, "models", "mnist.onnx")
    
    # # get voice command
    # recorder = NaoVoiceCommand(model_dir=llm_dir)
    # robot_command = recorder.record_audio()
    # recorder.close()

    robot_command = json.dumps({"wakeword": True, "action":"change_eye_color", "params":{"color":"red"}})
    
    # execute on robot
    cmd = [os.path.join(docker_dir, "run-naoqi.sh"), "python2.7", "src/nao_bundle/execute.py", ROBOT_IP, PORT]
    result = subprocess.run(cmd, input=robot_command, capture_output=True, text=True, check=False)
     
    # results
    print("returncode:", result.returncode)
    print("stdout:\n", result.stdout)
    print("stderr:\n", result.stderr)

if __name__ == "__main__":
    main()