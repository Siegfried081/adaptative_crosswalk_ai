import os
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "object-detection-occlusion"
PROJECT = "wheelchair-detection-in5sp"
VERSION = 1
FORMAT = "yolov8"


def download() -> Path:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY environment variable not set")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(FORMAT)

    data_yaml = Path(dataset.location) / "data.yaml"
    Path(".dataset_location").write_text(str(data_yaml))
    print(f"Dataset downloaded to: {dataset.location}")
    return data_yaml


if __name__ == "__main__":
    download()
