from ultralytics import YOLO
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    opt = parser.parse_args()

    model = YOLO(opt.weights)
    model.eval()

    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model.model, dummy_input, "yolo11n_v6.onnx", opset_version=12, input_names=['images'], output_names=['output'])
