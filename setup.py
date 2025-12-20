import os
import urllib.request
from huggingface_hub import snapshot_download

LLM_DIR = "./models/qwen"
YOLO_DIR = "./models/yolov8"
MNIST_PATH = "./models/mnist.onnx"
MNIST_URL = ("https://raw.githubusercontent.com/onnx/models/main/" "validated/vision/classification/mnist/model/mnist-8.onnx")

if not os.path.exists(LLM_DIR):
    local_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir=LLM_DIR, local_dir_use_symlinks=False)
    print("Model downloaded to:", local_dir)
if not os.path.exists(YOLO_DIR):
    local_dir = snapshot_download("ultralytics/yolov8", local_dir=YOLO_DIR, local_dir_use_symlinks=False)
    print("YOLO model downloaded to:", YOLO_DIR)
if not os.path.exists(MNIST_PATH):
    urllib.request.urlretrieve(MNIST_URL, MNIST_PATH)
    print("MNIST model downloaded to:", MNIST_PATH)