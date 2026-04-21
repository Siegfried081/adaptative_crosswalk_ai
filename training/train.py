from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/train.yaml")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_config(config: dict) -> None:
    required_keys = [
        "model",
        "data",
        "epochs",
        "batch",
        "imgsz",
        "project",
        "name",
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys in train.yaml: {missing_keys}")

    data_path = Path(config["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {data_path}")


def train() -> None:
    config = load_config(CONFIG_PATH)
    validate_config(config)

    print("Starting training with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    model = YOLO(config["model"])

    model.train(
        data=config["data"],
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        project=config["project"],
        name=config["name"],
        device=config.get("device", "cpu"),
        workers=config.get("workers", 2),
        pretrained=config.get("pretrained", True),
        save=config.get("save", True),
        verbose=config.get("verbose", True),
    )

    print("Training finished successfully.")


if __name__ == "__main__":
    train()
