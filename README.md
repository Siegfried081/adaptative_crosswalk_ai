# Adaptive Crosswalk AI

Computer vision module for the Adaptive Crosswalk project. Detects wheelchairs in real time using YOLOv8, deployed on a Raspberry Pi.

## Overview

This repository does not run training — it manages **model versioning and validation**. Training is performed externally (Google Colab), and the resulting model is submitted here as a challenger to be compared against the current production model (champion).

## Pipeline

```
Colab (training)
      │
      ▼
models/challenger.pt  ──► git push
      │
      ▼
CI: Code Quality
  - Black formatting
  - Flake8 lint
  - Unit tests
      │
      ▼
CI: Model Validation
  - Downloads test dataset from Roboflow (v1)
  - Evaluates challenger vs champion (mAP50, precision, recall)
  - Approves or rejects the challenger
      │
      ├── APPROVED ──► promote challenger to champion
      │
      └── REJECTED ──► champion remains, CI fails with metrics report
```

## Repository Structure

```
configs/
  train.yaml        # Training parameters (reference for Colab)
  inference.yaml    # Inference parameters for Raspberry Pi
  eval.yaml         # Validation parameters (thresholds, model paths)
models/
  challenger.pt     # New model submitted for evaluation
  champion.pt       # Current production model
scripts/
  compare_models.py # Evaluates and compares both models on the test set
  download_dataset.py # Downloads test split from Roboflow
src/
  main.py           # Inference entry point (runs on Raspberry Pi)
training/
  train.py          # Training script (reference for Colab execution)
  export_model.py   # Exports model to ONNX format
dataset/
  data.yaml         # Dataset configuration
```

## Workflow

### 1. Train a new model (Colab)

Use `configs/train.yaml` as reference for training parameters. After training, save the best weights:

```python
import shutil
shutil.copy("/content/runs/detect/train/weights/best.pt", "/content/drive/MyDrive/challenger.pt")
```

### 2. Submit the challenger

Copy the downloaded `challenger.pt` to the `models/` directory and push:

```bash
git add models/challenger.pt
git commit -m "feat: add challenger model"
git push
```

### 3. CI validation

The pipeline runs automatically. The `validate` job downloads the test dataset from Roboflow and compares both models. Results are available as a workflow artifact (`eval-report`) in GitHub Actions.

### 4. Promote to champion

If the CI passes, promote the challenger:

```bash
copy models\challenger.pt models\champion.pt
git add models\champion.pt
git commit -m "chore: promote challenger to champion"
git push
```

## Configuration

### Validation parameters (`configs/eval.yaml`)

| Parameter | Description |
|---|---|
| `challenger_model` | Path to the challenger model |
| `champion_model` | Path to the champion model |
| `conf` | Confidence threshold for evaluation |
| `min_map50_improvement` | Minimum mAP50 gain required to approve the challenger |

### Training parameters (`configs/train.yaml`)

| Parameter | Description |
|---|---|
| `model` | Base YOLOv8 model (e.g. `yolov8n.pt`) |
| `epochs` | Number of training epochs |
| `batch` | Batch size |
| `imgsz` | Input image size |

## Setup

```bash
pip install -r requirements.txt      # runtime
pip install -r requirements-dev.txt  # development (lint, tests)
```

### Required GitHub Secret

| Secret | Description |
|---|---|
| `ROBOFLOW_API_KEY` | Roboflow API key for downloading the test dataset |

## Inference (Raspberry Pi)

Edit `configs/inference.yaml` to point to the champion model, then run:

```bash
python src/main.py
```
