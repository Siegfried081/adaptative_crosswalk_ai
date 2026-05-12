import json
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/eval.yaml")
REPORTS_DIR = Path("reports")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    required_keys = ["challenger_model", "champion_model", "data", "conf"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required keys in eval.yaml: {missing}")

    data_path = Path(config["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")


def evaluate_model(
    model_path: str, data: str, conf: float, device: str = "cpu"
) -> dict:
    model = YOLO(model_path)
    start = time.perf_counter()
    results = model.val(
        data=data, conf=conf, device=device, split="test", verbose=False
    )
    elapsed = time.perf_counter() - start
    return {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "inference_time_s": round(elapsed, 2),
    }


def compare(
    challenger: dict, champion: dict, min_improvement: float = 0.0
) -> bool:
    return challenger["map50"] >= champion["map50"] + min_improvement


def print_report(challenger: dict, champion: dict, promoted: bool) -> None:
    print("\n=== Model Evaluation Report ===")
    header = f"{'Metric':<20} {'Champion':>12} {'Challenger':>12} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for metric in ["map50", "map50_95", "precision", "recall"]:
        c_val = champion.get(metric, 0.0)
        ch_val = challenger.get(metric, 0.0)
        delta = ch_val - c_val
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<20} {c_val:>12.4f} {ch_val:>12.4f} {sign}{delta:>9.4f}")
    print(
        f"\nInference time  champion={champion['inference_time_s']:.2f}s  "
        f"challenger={challenger['inference_time_s']:.2f}s"
    )
    verdict = "APPROVED" if promoted else "REJECTED"
    print(f"\nVerdict: {verdict}")


def save_report(report: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


def run() -> None:
    config = load_config()
    validate_config(config)

    challenger_path = Path(config["challenger_model"])
    champion_path = Path(config["champion_model"])
    data = config["data"]
    conf = config["conf"]
    device = config.get("device", "cpu")
    min_improvement = config.get("min_map50_improvement", 0.0)

    if not challenger_path.exists():
        raise FileNotFoundError(f"Challenger model not found: {challenger_path}")

    if not champion_path.exists():
        print(
            f"No champion model found at {champion_path}. "
            "Challenger auto-promoted as first champion."
        )
        return

    print(f"Evaluating champion: {champion_path}")
    champion = evaluate_model(str(champion_path), data, conf, device)

    print(f"Evaluating challenger: {challenger_path}")
    challenger = evaluate_model(str(challenger_path), data, conf, device)

    promoted = compare(challenger, champion, min_improvement)
    print_report(challenger, champion, promoted)
    save_report({"champion": champion, "challenger": challenger, "promoted": promoted})

    if not promoted:
        print(
            f"\nChallenger map50={challenger['map50']:.4f} did not beat "
            f"champion map50={champion['map50']:.4f} "
            f"(required improvement: +{min_improvement:.4f})."
        )
        sys.exit(1)


if __name__ == "__main__":
    run()
