from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/inference.yaml")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_config(config: dict) -> None:
    required_keys = ["model_path", "source", "conf"]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required config keys in inference.yaml: {missing_keys}"
        )

    model_path = Path(config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Inference model not found: {model_path}")


def parse_source(source_value):
    if source_value == "0":
        return 0
    return source_value


def run_inference() -> None:
    config = load_config(CONFIG_PATH)
    validate_config(config)

    model_path = config["model_path"]
    source = parse_source(config["source"])
    conf = config["conf"]
    save = config.get("save", True)
    show = config.get("show", False)
    device = config.get("device", "cpu")

    print("Starting inference with configuration:")
    print(f"  model_path: {model_path}")
    print(f"  source: {source}")
    print(f"  conf: {conf}")
    print(f"  save: {save}")
    print(f"  show: {show}")
    print(f"  device: {device}")

    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=show,
        device=device,
    )

    print("Inference finished successfully.")
    print(f"Processed results: {len(results)} item(s)")


if __name__ == "__main__":
    run_inference()
