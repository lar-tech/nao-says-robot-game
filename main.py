import os
import json
import subprocess

from huggingface_hub import snapshot_download

from src.voice_commands import NaoVoiceCommand
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # config
    ROBOT_IP = "192.168.1.118"
    PORT = "9559"
    MODEL_DIR = "./models/qwen"
    if not os.path.exists(MODEL_DIR):
        local_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir=MODEL_DIR, local_dir_use_symlinks=False)
        print("Model downloaded to:", local_dir)

    # get voice command
    with NaoVoiceCommand() as recorder:
        command = recorder.record_audio()
    payload = json.dumps(command, ensure_ascii=True)

    # execute on robot
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [parent_dir + "/run-naoqi.sh", "python2.7", "src/execute.py", ROBOT_IP, PORT, payload]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    # results
    print("returncode:", result.returncode)
    print("stdout:\n", result.stdout)
    print("stderr:\n", result.stderr)

