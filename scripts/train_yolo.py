#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fine-tune and compare COCO-pretrained Ultralytics detectors on CODD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "artifacts" / "yolo_dataset"
DEFAULT_RUNS = PROJECT_ROOT / "runs" / "detect"
PINNED_ULTRALYTICS_VERSION = "8.4.210"
SUPPORTED_MODELS = tuple(
    f"{family}{size}"
    for family in ("yolov8", "yolo11", "yolo26")
    for size in ("n", "s", "m", "l", "x")
)
OPTIMIZERS = ("SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto")


def json_safe(value: Any) -> Any:
    """Convert NumPy and other scalar values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_validated_dataset(dataset_dir: Path) -> tuple[Path, dict[str, Any]]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit("YOLO data.yaml is missing; run prepare_yolo_dataset.py first.")

    validation_path = dataset_dir / "validation_report.json"
    if not validation_path.is_file():
        raise SystemExit("validation_report.json is missing; run validate_yolo_dataset.py first.")
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit("validation_report.json is not valid JSON.") from exc
    if validation.get("passed") is not True:
        raise SystemExit("The YOLO dataset did not pass validation; fix the reported errors before training.")
    if validation.get("total_images", 0) <= 0 or validation.get("total_objects", 0) <= 0:
        raise SystemExit("The validation report contains no images or objects.")
    return data_yaml, validation


def check_ultralytics_compatibility() -> tuple[Any, Any, str, str]:
    """Require the pinned package version and return YOLO, torch, and versions."""
    try:
        installed_version = version("ultralytics")
    except PackageNotFoundError as exc:
        raise SystemExit("Ultralytics is not installed; run python -m pip install -r requirements.txt.") from exc
    if installed_version != PINNED_ULTRALYTICS_VERSION:
        raise SystemExit(
            f"This script targets ultralytics=={PINNED_ULTRALYTICS_VERSION}, "
            f"but {installed_version} is installed. Install the pinned requirements before training."
        )

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics or PyTorch could not be imported; reinstall requirements.txt.") from exc
    return YOLO, torch, installed_version, str(torch.__version__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune one or more COCO-pretrained Ultralytics detection models on CODD."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=["yolov8n"],
        metavar="MODEL",
        help="Model checkpoints to compare (default: yolov8n).",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET, help="Validated YOLO dataset directory")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS, help="Directory for Ultralytics run outputs")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Square training and validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size; -1 enables Ultralytics auto batch")
    parser.add_argument("--val-batch", type=int, default=None, help="Independent validation batch; defaults safely when batch=-1")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs")
    parser.add_argument("--workers", type=int, default=4, help="Data-loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed shared across model runs")
    parser.add_argument("--device", default=None, help="Training device, for example 0, cpu, or mps")
    parser.add_argument("--name-prefix", default="codd", help="Prefix for experiment directories")

    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="AdamW", help="Ultralytics optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning-rate fraction")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Optimizer weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup duration in epochs")
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="cosine", help="Learning-rate schedule")
    parser.add_argument("--freeze", type=int, default=0, help="Number of initial layers to freeze; 0 fine-tunes all layers")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable automatic mixed precision")

    # Explicit, modest image augmentations. The same values are used for every selected model.
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.imgsz < 32 or args.patience < 0 or args.workers < 0:
        raise SystemExit("epochs/imgsz must be positive and patience/workers cannot be negative.")
    if args.batch == 0 or args.batch < -1:
        raise SystemExit("batch must be -1 (auto) or a positive integer.")
    if args.val_batch is not None and args.val_batch < 1:
        raise SystemExit("val-batch must be a positive integer.")
    if args.freeze < 0 or args.warmup_epochs < 0:
        raise SystemExit("freeze and warmup-epochs cannot be negative.")
    if args.lr0 <= 0 or not 0 < args.lrf <= 1 or args.weight_decay < 0:
        raise SystemExit("lr0 must be positive, lrf must be in (0, 1], and weight-decay cannot be negative.")
    bounded_values = {
        "hsv-h": args.hsv_h,
        "hsv-s": args.hsv_s,
        "hsv-v": args.hsv_v,
        "translate": args.translate,
        "scale": args.scale,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
    }
    for name, value in bounded_values.items():
        if not 0 <= value <= 1:
            raise SystemExit(f"{name} must be between 0 and 1.")
    if args.degrees < 0 or args.shear < 0:
        raise SystemExit("degrees and shear cannot be negative.")
    if not args.name_prefix.strip():
        raise SystemExit("name-prefix cannot be empty.")


def hardware_info(torch: Any, requested_device: str | None) -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "requested_device": requested_device or "auto",
    }
    hardware["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        hardware["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
    hardware["mps_available"] = bool(mps)
    if requested_device == "cpu":
        hardware["accelerator_used"] = "cpu"
    elif requested_device == "mps" or (requested_device is None and mps and not torch.cuda.is_available()):
        hardware["accelerator_used"] = "mps"
    elif torch.cuda.is_available():
        hardware["accelerator_used"] = "cuda"
    else:
        hardware["accelerator_used"] = "cpu"
    return hardware


def training_history(results_csv: Path, trainer: Any) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8") as stream:
            rows = [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(stream)]

    first_epoch = None
    for row in rows:
        try:
            first_epoch = int(float(row.get("epoch", "nan")))
            break
        except (TypeError, ValueError):
            continue
    zero_based_epochs = first_epoch == 0

    best_score = float("-inf")
    best_epoch: int | None = None
    for row in rows:
        try:
            score = float(row.get("metrics/mAP50-95(B)", "nan"))
            epoch_number = int(float(row.get("epoch", "nan")))
        except (TypeError, ValueError):
            continue
        if score == score and score > best_score:  # Ignore NaN and retain the first tie.
            best_score = score
            best_epoch = epoch_number + 1 if zero_based_epochs else epoch_number

    last_epoch = getattr(trainer, "epoch", None)
    epochs_executed = len(rows) or (int(last_epoch) + 1 if last_epoch is not None else None)
    return {
        "epochs_executed": epochs_executed,
        "best_epoch": best_epoch,
        "best_epoch_selection_metric": "metrics/mAP50-95(B)",
        "best_epoch_metric_value": None if best_epoch is None else best_score,
    }


def effective_config(trainer: Any, args: argparse.Namespace, val_batch: int) -> dict[str, Any]:
    trainer_args = getattr(trainer, "args", None)
    config = dict(vars(trainer_args)) if trainer_args is not None and hasattr(trainer_args, "__dict__") else {}
    for key in ("data",):
        if key in config:
            config[key] = Path(str(config[key])).name
    for key in ("project", "save_dir"):
        if key in config and config[key] is not None:
            config[key] = Path(str(config[key])).name
    config.update({
        "scheduler": args.scheduler,
        "validation_batch": val_batch,
        "validation_split": "val",
        "test_split_used": False,
    })
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        config["optimizer_effective_class"] = type(optimizer).__name__
    if args.optimizer == "auto":
        config["learning_rate_note"] = "Ultralytics auto optimizer selects lr0; --lr0 is not applied as a fixed value."
    return json_safe(config)


def run_model(
    model_name: str,
    data_yaml: Path,
    dataset_validation: dict[str, Any],
    args: argparse.Namespace,
    YOLO: Any,
    torch: Any,
    ultralytics_version: str,
    torch_version: str,
) -> dict[str, Any]:
    # Only official model identifiers are accepted, never a path into CODD's weights folder.
    checkpoint_name = f"{model_name}.pt"
    run_name = f"{args.name_prefix}-{model_name}-img{args.imgsz}-seed{args.seed}"
    val_batch = args.val_batch if args.val_batch is not None else (args.batch if args.batch > 0 else 1)

    print(f"\n=== Training {checkpoint_name} on CODD ===")
    model = YOLO(checkpoint_name)
    started = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        freeze=args.freeze,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.scheduler == "cosine",
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        amp=args.amp,
        val=True,
        split="val",
        plots=True,
        save=True,
        project=str(args.project),
        name=run_name,
        exist_ok=False,
        device=args.device,
        verbose=True,
    )
    training_seconds = time.perf_counter() - started

    trainer = getattr(model, "trainer", None)
    run_dir = Path(trainer.save_dir) if trainer and getattr(trainer, "save_dir", None) else args.project / run_name
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce best.pt for {run_name}.")

    # Re-evaluate the selected checkpoint on validation only. Never pass test to train() or val().
    best_model = YOLO(str(best_weights))
    validation_started = time.perf_counter()
    validation_metrics = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=val_batch,
        workers=args.workers,
        project=str(args.project / "validation"),
        name=run_dir.name,
        exist_ok=False,
        plots=True,
        device=args.device,
        verbose=True,
    )
    validation_seconds = time.perf_counter() - validation_started

    speed = json_safe(getattr(validation_metrics, "speed", {}) or {})
    inference_ms = speed.get("inference")
    try:
        inference_fps = 1000.0 / float(inference_ms) if float(inference_ms) > 0 else None
    except (TypeError, ValueError):
        inference_fps = None

    result = {
        "status": "completed",
        "model": model_name,
        "checkpoint": checkpoint_name,
        "checkpoint_source": f"Ultralytics official COCO-pretrained {model_name} detection weights",
        "fine_tuning": {
            "pretrained": True,
            "frozen_layers": args.freeze,
        },
        "dataset": dataset_dir_name(dataset_validation),
        "dataset_images": dataset_validation["total_images"],
        "dataset_objects": dataset_validation["total_objects"],
        "training": {
            "epochs_max": args.epochs,
            **training_history(run_dir / "results.csv", trainer),
            "training_seconds": round(training_seconds, 3),
            "validation_seconds": round(validation_seconds, 3),
            "imgsz": args.imgsz,
            "batch_requested": args.batch,
            "batch_validation": val_batch,
            "patience": args.patience,
            "workers": args.workers,
            "seed": args.seed,
            "deterministic": True,
            "device": args.device or "auto",
            "validation_split": "val",
            "test_split_used": False,
        },
        "configuration": effective_config(trainer, args, val_batch),
        "hardware": hardware_info(torch, args.device),
        "software": {
            "python": platform.python_version(),
            "ultralytics": ultralytics_version,
            "torch": torch_version,
        },
        "run_directory": run_dir.name,
        "best_weights": f"{run_dir.name}/weights/best.pt",
        "validation_metrics": json_safe(validation_metrics.results_dict),
        "validation_per_class": json_safe(validation_metrics.summary()),
        "validation_speed_ms_per_image": speed,
        "inference_latency_ms_per_image": inference_ms,
        "inference_images_per_second": inference_fps,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "experiment_summary.json", result)
    print(f"Saved experiment summary: {run_dir.name}/experiment_summary.json")
    return result


def dataset_dir_name(validation: dict[str, Any]) -> str:
    # Keep manifests portable: the validation report stores the dataset name, not a local path.
    return str(validation.get("dataset", "yolo_dataset"))


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    data_yaml, dataset_validation = load_validated_dataset(dataset_dir)
    YOLO, torch, ultralytics_version, torch_version = check_ultralytics_compatibility()

    comparison: dict[str, Any] = {
        "dataset": dataset_dir.name,
        "data_yaml": data_yaml.name,
        "dataset_validation_passed": True,
        "ultralytics_version": ultralytics_version,
        "models": [],
    }
    failures = 0
    args.project.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        try:
            result = run_model(
                model_name, data_yaml, dataset_validation, args, YOLO, torch, ultralytics_version, torch_version
            )
        except Exception as exc:
            failures += 1
            result = {
                "status": "failed",
                "model": model_name,
                "checkpoint": f"{model_name}.pt",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"Training {model_name} failed: {type(exc).__name__}: {exc}")
        comparison["models"].append(json_safe(result))
        write_json(args.project / "model_comparison.json", comparison)

    completed = len(args.models) - failures
    print(f"\nCompleted {completed}/{len(args.models)} model runs.")
    print(f"Comparison summary: {args.project / 'model_comparison.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
