from ultralytics import YOLO


def export_model():
    model = YOLO("runs/train/wheelchair_detector/weights/best.pt")
    model.export(format="onnx")


if __name__ == "__main__":
    export_model()
