import os
import urllib.request
from huggingface_hub import snapshot_download

def main():
    # paths and urls
    os.makedirs("./models", exist_ok=True)
    llm_dir = "./models/qwen"
    yolo_path = "./models/yolov8n.pt"
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    mnist_path = "./models/mnist.onnx"
    mnist_url = ("https://raw.githubusercontent.com/onnx/models/main/" "validated/vision/classification/mnist/model/mnist-8.onnx")

    # qwen
    if not os.path.exists(llm_dir):
        print("Downloading Qwen LLM...")
        local_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir=llm_dir, local_dir_use_symlinks=False)
    else:
        print("Qwen LLM already present.")

    # yolo
    if not os.path.exists(yolo_path):
        print("Downloading YOLO model...")
        urllib.request.urlretrieve(yolo_url, yolo_path)
    else:
        print("YOLO model already present.")

    # mnist
    if not os.path.exists(mnist_path):
        print("Downloading MNIST model...")
        urllib.request.urlretrieve(mnist_url, mnist_path)
    else:
        print("MNIST model already present.")
