import torch
from ultralytics import YOLO
def train():
    d = "0" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = YOLO("yolov8x.pt")
    model.train(data="coco.yaml", epochs=100, imgsz=640, batch=16, device=d, amp=True)
    model.export(format="onnx", dynamic=True)
if __name__ == "__main__": train()