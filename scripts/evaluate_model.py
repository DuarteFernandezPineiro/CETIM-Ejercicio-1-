#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate a trained Ultralytics detector on CODD without leaking the test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "artifacts" / "yolo_dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "model_evaluation"
PINNED_ULTRALYTICS_VERSION = "8.4.210"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOCALIZATION_IOU = 0.1


class _SerialThreadPool:
    """Small synchronous fallback for Ultralytics' initial label-cache scan."""

    def __init__(self, processes: int | None = None) -> None:
        self.processes = processes

    def __enter__(self) -> "_SerialThreadPool":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None

    def imap(self, func: Any, iterable: Iterable[Any]) -> Iterable[Any]:
        return map(func, iterable)


@contextmanager
def _serial_ultralytics_label_scan() -> Iterable[None]:
    """Avoid multiprocessing-pipe permissions during a first cache scan on Windows."""
    try:
        import ultralytics.data.dataset as dataset_module
    except ImportError:
        yield
        return
    original_pool = dataset_module.ThreadPool
    dataset_module.ThreadPool = _SerialThreadPool
    try:
        yield
    finally:
        dataset_module.ThreadPool = original_pool


def json_safe(value: Any) -> Any:
    """Convert NumPy and other scalar values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON artifact, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for XYXY boxes, including correctly shaped empty arrays."""
    a = np.asarray(boxes_a, dtype=float).reshape((-1, 4))
    b = np.asarray(boxes_b, dtype=float).reshape((-1, 4))
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection_wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def size_bin(box: Iterable[float], image_width: int, image_height: int) -> str:
    """Bucket a ground-truth box by image-area fraction using fixed audit-informed cutoffs."""
    x1, y1, x2, y2 = (float(value) for value in box)
    image_area = max(float(image_width * image_height), 1.0)
    fraction = max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / image_area
    if fraction <= 0.01:
        return "small"
    if fraction <= 0.10:
        return "medium"
    return "large"


def density_bin(object_count: int) -> str:
    """Bucket a scene by object count using the dataset audit's distribution landmarks."""
    if object_count <= 5:
        return "sparse_1_5"
    if object_count <= 9:
        return "medium_6_9"
    return "dense_10_plus"


def overlap_bins(gt_boxes: np.ndarray) -> list[str]:
    """Classify each GT by its maximum IoU with another GT in the same image."""
    boxes = np.asarray(gt_boxes, dtype=float).reshape((-1, 4))
    overlaps = box_iou_matrix(boxes, boxes)
    if len(boxes):
        np.fill_diagonal(overlaps, 0.0)
    labels: list[str] = []
    for row in overlaps:
        maximum = float(row.max()) if len(row) else 0.0
        if maximum < 0.1:
            labels.append("low_lt_0_1")
        elif maximum < 0.5:
            labels.append("moderate_0_1_0_5")
        else:
            labels.append("high_ge_0_5")
    return labels


def match_image(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_confidences: np.ndarray,
    match_iou: float = 0.5,
    localization_iou: float = LOCALIZATION_IOU,
) -> dict[str, Any]:
    """Greedily match predictions to GT once each, in descending confidence order.

    Same-class matches above ``match_iou`` are true positives. Remaining predictions
    can claim one unmatched GT as a classification error (IoU >= match_iou), a
    localization error (localization_iou <= IoU < match_iou), or a false positive.
    Unmatched GT boxes are false negatives. This diagnostic matching is not used for AP.
    """
    gt = np.asarray(gt_boxes, dtype=float).reshape((-1, 4))
    gt_cls = np.asarray(gt_classes, dtype=int).reshape((-1,))
    pred = np.asarray(pred_boxes, dtype=float).reshape((-1, 4))
    pred_cls = np.asarray(pred_classes, dtype=int).reshape((-1,))
    confidence = np.asarray(pred_confidences, dtype=float).reshape((-1,))
    ious = box_iou_matrix(pred, gt)
    unmatched_gt = set(range(len(gt)))
    matched: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    true_positive_gt: set[int] = set()

    for pred_index in np.argsort(-confidence, kind="stable"):
        available = sorted(unmatched_gt)
        if not available:
            errors.append({"type": "false_positive", "pred_index": int(pred_index), "gt_index": None, "iou": 0.0})
            continue

        same_class = [index for index in available if gt_cls[index] == pred_cls[pred_index]]
        best_same = max(same_class, key=lambda index: ious[pred_index, index], default=None)
        if best_same is not None and ious[pred_index, best_same] >= match_iou:
            unmatched_gt.remove(best_same)
            true_positive_gt.add(best_same)
            matched.append({"pred_index": int(pred_index), "gt_index": best_same, "iou": float(ious[pred_index, best_same])})
            continue

        best_any = max(available, key=lambda index: ious[pred_index, index])
        best_iou = float(ious[pred_index, best_any])
        if best_iou >= match_iou:
            unmatched_gt.remove(best_any)
            errors.append({
                "type": "classification_error", "pred_index": int(pred_index),
                "gt_index": best_any, "iou": best_iou,
            })
        elif best_iou >= localization_iou:
            unmatched_gt.remove(best_any)
            errors.append({
                "type": "localization_error", "pred_index": int(pred_index),
                "gt_index": best_any, "iou": best_iou,
            })
        else:
            errors.append({"type": "false_positive", "pred_index": int(pred_index), "gt_index": None, "iou": best_iou})

    for gt_index in sorted(unmatched_gt):
        errors.append({"type": "false_negative", "pred_index": None, "gt_index": gt_index, "iou": 0.0})

    return {"matches": matched, "errors": errors, "true_positive_gt": true_positive_gt}


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Compute a ratio, returning None when the metric is undefined."""
    return float(numerator) / float(denominator) if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    """Calculate harmonic mean at the same fixed confidence and IoU operating point."""
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision == 0 or recall == 0 else None
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    """Define the evaluation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained YOLO checkpoint, usually best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Evaluation split (default: val)")
    parser.add_argument("--confirm-test", action="store_true", help="Required acknowledgement to evaluate the reserved test split")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET, help="Validated YOLO dataset directory")
    parser.add_argument("--imgsz", type=int, default=None, help="Input size; defaults to the training configuration when found")
    parser.add_argument("--batch", type=int, default=None, help="Inference/validation batch; defaults to training val batch or 1")
    parser.add_argument("--device", default=None, help="Ultralytics device (e.g. 0 or cpu; default: auto)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Base directory for unique evaluation outputs")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for fixed-point error analysis")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for diagnostic one-to-one matches")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold for error-analysis predictions")
    parser.add_argument("--max-examples", type=int, default=4, help="Maximum visual examples per outcome type")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe split use and invalid numeric ranges before loading a model."""
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("The test split is reserved. Re-run with --split test --confirm-test only after model selection.")
    if args.checkpoint.suffix.lower() != ".pt" or not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist or is not a .pt file: {args.checkpoint}")
    if args.imgsz is not None and args.imgsz < 32:
        raise SystemExit("imgsz must be at least 32 pixels.")
    if args.batch is not None and args.batch < 1:
        raise SystemExit("batch must be a positive integer for independent evaluation.")
    if not 0 <= args.conf <= 1 or not 0 <= args.match_iou <= 1 or not 0 <= args.nms_iou <= 1:
        raise SystemExit("conf, match-iou, and nms-iou must be in [0, 1].")
    if args.max_examples < 0:
        raise SystemExit("max-examples cannot be negative.")


def _training_config(checkpoint: Path) -> dict[str, Any]:
    """Load nearby experiment metadata without requiring it to evaluate a checkpoint."""
    run_dir = checkpoint.parent.parent
    summary_path = run_dir / "experiment_summary.json"
    config: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            config.update(summary.get("training", {}))
            config.update({key: value for key, value in summary.get("configuration", {}).items() if key not in config})
        except (OSError, json.JSONDecodeError):
            pass
    args_path = run_dir / "args.yaml"
    if args_path.is_file():
        try:
            raw = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
            for key, value in raw.items():
                config.setdefault(key, value)
        except (OSError, yaml.YAMLError):
            pass
    return config


def _class_names(dataset_dir: Path, data_yaml: Path) -> list[str]:
    """Read class names from the validator report, falling back to data.yaml."""
    report_path = dataset_dir / "validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise SystemExit("The converted dataset has not passed validation.")
        classes = report.get("classes")
        if isinstance(classes, list) and classes:
            return [str(name) for name in classes]
    yaml_data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = yaml_data.get("names", {})
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    raise SystemExit("Class names were not found in validation_report.json or data.yaml.")


def _sanitize_native_args(run_dir: Path) -> None:
    """Keep Ultralytics' native output portable by stripping local absolute paths."""
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return
    try:
        values = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if isinstance(value, str) and (Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)):
            values[key] = Path(value).name
    args_path.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read normalized YOLO labels and convert them to pixel XYXY boxes."""
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file for evaluation image: {label_path.name}")
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path.name}:{line_number}")
        class_id, x_center, y_center, box_width, box_height = map(float, fields)
        cx, cy, bw, bh = x_center * width, y_center * height, box_width * width, box_height * height
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        classes.append(int(class_id))
    return np.asarray(boxes, dtype=float).reshape((-1, 4)), np.asarray(classes, dtype=int)


def _duplicate_label_rows(label_dir: Path, class_names: list[str]) -> list[dict[str, Any]]:
    """Report exact duplicate YOLO rows without changing source labels."""
    duplicates: list[dict[str, Any]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts = Counter(rows)
        for row, count in counts.items():
            if count <= 1:
                continue
            class_id = int(row.split()[0])
            duplicates.append({
                "image": f"{label_path.stem}.jpg",
                "class": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                "duplicate_rows": count - 1,
            })
    return duplicates


def _prediction_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one Ultralytics Results object into CPU NumPy arrays."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,))
    return (
        boxes.xyxy.detach().cpu().numpy().astype(float),
        boxes.cls.detach().cpu().numpy().astype(int),
        boxes.conf.detach().cpu().numpy().astype(float),
    )


def predict_in_batches(
    model: Any, image_paths: list[Path], batch: int, **options: Any
) -> Iterable[Any]:
    """Bound the number of images passed to Ultralytics' in-memory list loader."""
    for offset in range(0, len(image_paths), batch):
        paths = [str(path) for path in image_paths[offset:offset + batch]]
        yield from model.predict(source=paths, stream=True, batch=batch, **options)


def check_model_classes(model: Any, class_names: list[str]) -> None:
    expected = dict(enumerate(class_names))
    names = model.names
    if names != expected and names != class_names:
        raise SystemExit("Checkpoint class IDs or names do not match the dataset.")


def _metrics_by_class(native_summary: list[dict[str, Any]], class_names: list[str]) -> dict[str, dict[str, Any]]:
    """Normalize Ultralytics' per-class summary while preserving absent values."""
    by_name: dict[str, dict[str, Any]] = {}
    for row in native_summary:
        name = str(row.get("Class", row.get("class", "")))
        if not name:
            continue
        by_name[name] = row
    output: dict[str, dict[str, Any]] = {}
    for name in class_names:
        row = by_name.get(name, {})
        output[name] = {
            "images": row.get("Images"),
            "instances": row.get("Instances"),
            "ap50": row.get("mAP50"),
            "ap50_95": row.get("mAP50-95"),
        }
    return output


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _ratio_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    precision = safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
    return {"precision": precision, "recall": recall, "f1": f1_score(precision, recall)}


def _make_confusion_matrix(names: list[str], image_records: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an object-level confusion matrix with an explicit background row/column."""
    size = len(names) + 1
    background = len(names)
    matrix = np.zeros((size, size), dtype=np.int64)
    class_counts = {name: _empty_counts() for name in names}
    error_counts: Counter[str] = Counter()
    class_confusions: Counter[tuple[str, str]] = Counter()
    for record in image_records:
        gt_classes = record["gt_classes"]
        pred_classes = record["pred_classes"]
        matching = record["matching"]
        for match in matching["matches"]:
            gt_id = int(gt_classes[match["gt_index"]])
            pred_id = int(pred_classes[match["pred_index"]])
            matrix[gt_id, pred_id] += 1
            class_counts[names[gt_id]]["tp"] += 1
        for error in matching["errors"]:
            error_counts[error["type"]] += 1
            pred_index, gt_index = error["pred_index"], error["gt_index"]
            if error["type"] == "false_positive":
                pred_id = int(pred_classes[pred_index])
                matrix[background, pred_id] += 1
                class_counts[names[pred_id]]["fp"] += 1
            elif error["type"] == "false_negative":
                gt_id = int(gt_classes[gt_index])
                matrix[gt_id, background] += 1
                class_counts[names[gt_id]]["fn"] += 1
            elif error["type"] == "classification_error":
                gt_id = int(gt_classes[gt_index])
                pred_id = int(pred_classes[pred_index])
                matrix[gt_id, pred_id] += 1
                class_confusions[(names[gt_id], names[pred_id])] += 1
                class_counts[names[pred_id]]["fp"] += 1
                class_counts[names[gt_id]]["fn"] += 1
            elif error["type"] == "localization_error":
                gt_id = int(gt_classes[gt_index])
                pred_id = int(pred_classes[pred_index])
                matrix[gt_id, background] += 1
                matrix[background, pred_id] += 1
                class_counts[names[pred_id]]["fp"] += 1
                class_counts[names[gt_id]]["fn"] += 1
    confusion_list = [
        {"ground_truth_class": truth, "predicted_class": prediction, "count": count}
        for (truth, prediction), count in class_confusions.most_common()
    ]
    return matrix, {"errors": dict(error_counts), "class_counts": class_counts, "class_confusions": confusion_list}


def _error_rows(image_records: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    """Flatten image-level assignments into an auditable object-level error table."""
    output: list[dict[str, Any]] = []
    for record in image_records:
        for error in record["matching"]["errors"]:
            pred_index, gt_index = error["pred_index"], error["gt_index"]
            pred_box = record["pred_boxes"][pred_index].tolist() if pred_index is not None else None
            gt_box = record["gt_boxes"][gt_index].tolist() if gt_index is not None else None
            output.append({
                "image": record["path"].name,
                "error_type": error["type"],
                "ground_truth_class": class_names[int(record["gt_classes"][gt_index])] if gt_index is not None else None,
                "predicted_class": class_names[int(record["pred_classes"][pred_index])] if pred_index is not None else None,
                "confidence": float(record["pred_confidences"][pred_index]) if pred_index is not None else None,
                "iou": float(error["iou"]),
                "ground_truth_box_xyxy": json.dumps(gt_box) if gt_box is not None else None,
                "predicted_box_xyxy": json.dumps(pred_box) if pred_box is not None else None,
                "object_size_bin": record["gt_size_bins"][gt_index] if gt_index is not None else None,
                "scene_density_bin": record["density_bin"],
                "object_overlap_bin": record["gt_overlap_bins"][gt_index] if gt_index is not None else None,
            })
    return output


def _group_recall(image_records: list[dict[str, Any]], group_key: str, object_key: str | None = None) -> dict[str, dict[str, int | float | None]]:
    """Calculate recall by GT-defined subgroup, never calculating AP on a subset."""
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for record in image_records:
        bins = record[object_key] if object_key else [record[group_key]] * len(record["gt_classes"])
        for gt_index, label in enumerate(bins):
            totals[label] += 1
            if gt_index in record["matching"]["true_positive_gt"]:
                hits[label] += 1
    category_order = {
        "small": 0, "medium": 1, "large": 2,
        "sparse_1_5": 0, "medium_6_9": 1, "dense_10_plus": 2,
        "low_lt_0_1": 0, "moderate_0_1_0_5": 1, "high_ge_0_5": 2,
    }
    return {
        label: {"ground_truth_objects": int(totals[label]), "true_positives": int(hits[label]), "recall": safe_ratio(hits[label], totals[label])}
        for label in sorted(totals, key=lambda value: (category_order.get(value, 99), value))
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV, retaining nulls as empty cells."""
    fieldnames = list(rows[0]) if rows else ["image", "error_type"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(matrix: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    """Save absolute and row-normalized confusion matrices including background."""
    labels = [*class_names, "background"]
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    for values, filename, title, fmt in (
        (matrix, "confusion_matrix_absolute.png", "Confusion matrix (counts)", "d"),
        (normalized, "confusion_matrix_normalized.png", "Confusion matrix (row normalized)", ".2f"),
    ):
        width = max(9, len(labels) * 0.8)
        fig, ax = plt.subplots(figsize=(width, width * 0.85))
        image = ax.imshow(values, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_title(title)
        threshold = float(np.max(values)) / 2 if values.size else 0
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(column, row, format(value, fmt), ha="center", va="center", fontsize=7,
                        color="white" if value > threshold else "black")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def _plot_class_metrics(class_metrics: dict[str, dict[str, Any]], fixed_metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Plot AP50-95 and fixed-point F1 per class."""
    labels = list(class_metrics)
    positions = np.arange(len(labels))
    ap = [class_metrics[name].get("ap50_95") for name in labels]
    f1 = [fixed_metrics[name].get("f1") for name in labels]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(positions - 0.2, [value if value is not None else np.nan for value in ap], width=0.4, label="AP50-95")
    ax.bar(positions + 0.2, [value if value is not None else np.nan for value in f1], width=0.4, label="F1 at fixed operating point")
    ax.set_xticks(positions, labels=labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title("Metrics by class")
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=160)
    plt.close(fig)


def _plot_group_recall(groups: dict[str, dict[str, Any]], filename: str, title: str, output_dir: Path) -> None:
    """Plot GT recall for predefined size, density, or overlap groups."""
    labels = list(groups)
    values = [groups[label]["recall"] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, [value if value is not None else 0 for value in values], color="#3976a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall at fixed confidence/IoU")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for index, label in enumerate(labels):
        count = groups[label]["ground_truth_objects"]
        ax.text(index, (values[index] or 0) + 0.02, f"n={count}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def _draw_comparison(image_path: Path, record: dict[str, Any], class_names: list[str], destination: Path) -> None:
    """Render side-by-side GT and prediction panels for one image."""
    with Image.open(image_path) as source:
        base = source.convert("RGB")
    panel_width, panel_height = base.size
    canvas = Image.new("RGB", (panel_width * 2, panel_height + 44), "white")
    canvas.paste(base, (0, 44))
    canvas.paste(base, (panel_width, 44))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 12), "Ground truth", fill="black", font=font)
    draw.text((panel_width + 10, 12), "Predictions", fill="black", font=font)
    for box, class_id in zip(record["gt_boxes"], record["gt_classes"]):
        coords = [int(round(value)) for value in box]
        coords[1] += 44
        coords[3] += 44
        name = class_names[int(class_id)]
        draw.rectangle(coords, outline="#20a060", width=4)
        draw.text((coords[0] + 2, max(44, coords[1] - 12)), name, fill="#20a060", font=font)
    for box, class_id, confidence in zip(record["pred_boxes"], record["pred_classes"], record["pred_confidences"]):
        coords = [int(round(value)) for value in box]
        coords[0] += panel_width
        coords[2] += panel_width
        coords[1] += 44
        coords[3] += 44
        label = f"{class_names[int(class_id)]} {float(confidence):.2f}"
        draw.rectangle(coords, outline="#df493c", width=4)
        draw.text((coords[0] + 2, max(44, coords[1] - 12)), label, fill="#df493c", font=font)
    canvas.thumbnail((1600, 1000))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=88)


def _write_visual_examples(image_records: list[dict[str, Any]], class_names: list[str], output_dir: Path, maximum: int) -> dict[str, list[str]]:
    """Save a capped number of correct, FP, and FN side-by-side examples."""
    saved: dict[str, list[str]] = {"correct": [], "false_positive": [], "false_negative": []}
    if maximum == 0:
        return saved
    for record in image_records:
        categories = set()
        if record["matching"]["matches"]:
            categories.add("correct")
        for error in record["matching"]["errors"]:
            if error["type"] in {"false_positive", "classification_error", "localization_error"}:
                categories.add("false_positive")
            if error["type"] in {"false_negative", "classification_error", "localization_error"}:
                categories.add("false_negative")
        for category in categories:
            if len(saved[category]) >= maximum:
                continue
            filename = f"{category}_{record['path'].stem}.jpg"
            relative = Path("visual_examples") / filename
            _draw_comparison(record["path"], record, class_names, output_dir / relative)
            saved[category].append(relative.as_posix())
        if all(len(paths) >= maximum for paths in saved.values()):
            break
    return saved


def _metric_value(metrics: Any, key: str) -> float | None:
    """Read an Ultralytics global metric by its stable display key."""
    results = getattr(metrics, "results_dict", {}) or {}
    value = results.get(key)
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _relative_or_name(path: Path) -> str:
    """Keep paths portable in artifacts by recording a project-relative path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _output_path(base: Path, checkpoint: Path, model_name: str, imgsz: int, split: str) -> Path:
    """Create a unique, non-overwriting evaluation directory."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-") or checkpoint.stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = base / f"{safe_name}-img{imgsz}-{split}-{timestamp}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def _hardware_info(torch: Any, requested_device: str | None) -> dict[str, Any]:
    """Describe host and visible accelerator without allocating a CUDA tensor."""
    info: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": __import__("os").cpu_count(),
        "requested_device": requested_device or "auto",
        "cuda_available": bool(torch.cuda.is_available()),
        "accelerator_used": "cpu" if requested_device == "cpu" or not torch.cuda.is_available() else "cuda",
        "cuda_devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            info["cuda_devices"].append({
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_gib": round(properties.total_memory / (1024**3), 3),
            })
    return info


def _write_report(path: Path, summary: dict[str, Any], output_dir: Path, class_names: list[str]) -> None:
    """Create a concise Markdown report with method and explicit limitations."""
    global_metrics = summary["global_metrics"]
    operating = summary["operating_point_metrics"]
    lines = [
        "# Model evaluation report",
        "",
        f"- Model: `{summary['model']}`",
        f"- Split: `{summary['split']}`",
        f"- Images: {summary['images_evaluated']}",
        f"- Raw label objects: {summary['ground_truth_objects']}",
        f"- Ground-truth objects scored by Ultralytics: {summary.get('native_ground_truth_objects', summary['ground_truth_objects'])}",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Input size / batch: {summary['imgsz']} / {summary['batch']}",
        "",
        "## Global metrics",
        "",
        "AP metrics come from Ultralytics' native validator. Precision, recall, and F1 are computed from the diagnostic one-to-one matches at the fixed operating point below.",
        "",
        f"- mAP50-95: {global_metrics.get('map50_95')}",
        f"- mAP50: {global_metrics.get('map50')}",
        f"- Precision: {operating.get('precision')}",
        f"- Recall: {operating.get('recall')}",
        f"- F1: {operating.get('f1')}",
        f"- Operating point: confidence >= {summary['thresholds']['confidence']}, one-to-one IoU >= {summary['thresholds']['match_iou']}; NMS IoU={summary['thresholds']['nms_iou']}.",
        "",
        "## Per-class results",
        "",
        "| Class | Instances | AP50 | AP50-95 | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in class_names:
        ap = summary["per_class_ap"].get(name, {})
        fixed = summary["per_class_operating_point"].get(name, {})
        lines.append(f"| {name} | {ap.get('instances')} | {ap.get('ap50')} | {ap.get('ap50_95')} | {fixed.get('precision')} | {fixed.get('recall')} | {fixed.get('f1')} |")
    lines.extend([
        "",
        "## Performance by object and scene",
        "",
        "Recall by size, scene density, and overlap uses ground-truth subsets at the same fixed confidence/IoU operating point. AP is not recomputed on subsets.",
        "",
        f"- Size rule: small <= 1% image area; medium > 1% and <= 10%; large > 10%. These fixed bins reflect the dataset audit, which found about half of boxes below 1% area.",
        f"- Density rule: sparse 1-5 objects, medium 6-9, dense >= 10; cutoffs reflect the audit's concentration around <=5 and >=10 objects per image.",
        f"- Overlap rule: maximum pairwise GT IoU < 0.1, 0.1-<0.5, or >=0.5. Single-object scenes fall in the low-overlap group.",
        "",
        f"- Size recall: `{summary['recall_by_size']}`",
        f"- Density recall: `{summary['recall_by_density']}`",
        f"- Overlap recall: `{summary['recall_by_overlap']}`",
        "",
        "## Errors and plots",
        "",
        f"- Object-level errors: `{summary['error_counts']}`",
        f"- Most frequent cross-class confusions: `{summary['class_confusions'][:10]}`",
        "- Error rows: `error_analysis.csv`",
        "- Confusion matrices: `confusion_matrix_absolute.png`, `confusion_matrix_normalized.png`",
        "- Class metrics: `per_class_metrics.png`",
        "- Group recall: `recall_by_size.png`, `recall_by_density.png`, `recall_by_overlap.png`",
        "- Native Ultralytics PR curves and validation plots: `ultralytics/native_validation/`",
        "- GT/prediction side-by-side examples: `visual_examples/`",
        "",
        "## Runtime and limitations",
        "",
        f"- Native validation wall time: {summary['runtime']['validation_wall_seconds']} seconds.",
        f"- Error-analysis prediction pass wall time: {summary['runtime']['error_analysis_pass_seconds']} seconds; this includes batching, preprocessing, inference, postprocessing, and result transfer.",
        f"- Native per-image speed breakdown: `{summary['runtime']['native_speed_ms_per_image']}` ms/image.",
        f"- Parameters: {summary.get('parameter_count')}",
        "- One-to-one matching is confidence-ordered greedy matching. It is used only for thresholded diagnostics; AP remains Ultralytics-native.",
        "- Group recall is descriptive and depends on the specified confidence, IoU, and fixed bins; it is not a replacement for mAP.",
    ])
    if summary.get("native_ground_truth_objects") != summary.get("ground_truth_objects"):
        differing = [
            f"{name}: native={values.get('instances')}, raw={values.get('raw_annotation_instances')}"
            for name, values in summary["per_class_ap"].items()
            if values.get("instances") != values.get("raw_annotation_instances")
        ]
        lines.extend([
            f"- Label-cache discrepancy: Ultralytics scored {summary['native_ground_truth_objects']} objects after removing duplicate label rows; raw files contain {summary['ground_truth_objects']}.",
            f"- Affected classes: {', '.join(differing)}.",
            "",
        ])
    duplicate_labels = summary.get("label_integrity", {}).get("exact_duplicate_label_rows", [])
    if duplicate_labels:
        lines.extend([
            f"- Exact duplicate YOLO label rows (reported, not edited): `{duplicate_labels}`",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> Path:
    """Run native validation plus fixed-threshold error diagnostics and write artifacts."""
    validate_args(args)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    data_yaml = dataset_dir / "data.yaml"
    validation_report_path = dataset_dir / "validation_report.json"
    if not data_yaml.is_file() or not validation_report_path.is_file():
        raise SystemExit("data.yaml and validation_report.json are required; prepare and validate the dataset first.")
    report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise SystemExit("The dataset did not pass validate_yolo_dataset.py.")
    class_names = _class_names(dataset_dir, data_yaml)
    checkpoint = args.checkpoint.expanduser().resolve()
    config = _training_config(checkpoint)
    imgsz = args.imgsz or int(config.get("imgsz", 640))
    batch = args.batch or int(config.get("batch_validation", config.get("validation_batch", config.get("batch", 1))))
    if batch < 1:
        batch = 1
    split_dir = dataset_dir / "images" / args.split
    label_dir = dataset_dir / "labels" / args.split
    if not split_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"Missing image/label directories for split '{args.split}'.")
    image_paths = sorted(path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"No supported images found for split '{args.split}'.")
    output_dir = _output_path(args.output_dir.expanduser().resolve(), checkpoint, str(config.get("model", checkpoint.parent.parent.name)), imgsz, args.split)
    duplicate_labels = _duplicate_label_rows(label_dir, class_names)

    try:
        installed = version("ultralytics")
    except PackageNotFoundError as exc:
        raise SystemExit("Ultralytics is not installed; install the project requirements first.") from exc
    if installed != PINNED_ULTRALYTICS_VERSION:
        raise SystemExit(f"Expected ultralytics=={PINNED_ULTRALYTICS_VERSION}, found {installed}.")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("PyTorch or Ultralytics could not be imported.") from exc

    model = YOLO(str(checkpoint))
    check_model_classes(model, class_names)
    parameters = None
    try:
        parameters = sum(parameter.numel() for parameter in model.model.parameters())
    except (AttributeError, TypeError):
        pass

    validation_started = time.perf_counter()
    with _serial_ultralytics_label_scan():
        native_metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=imgsz,
            batch=batch,
            conf=0.001,
            iou=0.7,
            workers=0,
            plots=True,
            project=str(output_dir / "ultralytics"),
            name="native_validation",
            exist_ok=False,
            device=args.device,
            verbose=True,
        )
    validation_seconds = time.perf_counter() - validation_started
    native_speed = json_safe(getattr(native_metrics, "speed", {}) or {})
    native_summary = json_safe(native_metrics.summary())
    native_save_dir = Path(getattr(native_metrics, "save_dir", output_dir / "ultralytics" / "native_validation"))
    _sanitize_native_args(native_save_dir)

    image_records: list[dict[str, Any]] = []
    prediction_started = time.perf_counter()
    prediction_stream = predict_in_batches(
        model,
        image_paths,
        imgsz=imgsz,
        batch=batch,
        conf=args.conf,
        iou=args.nms_iou,
        device=args.device,
        verbose=False,
        save=False,
    )
    seen_paths: set[Path] = set()
    for result in prediction_stream:
        image_path = Path(result.path).resolve()
        seen_paths.add(image_path)
        height, width = map(int, result.orig_shape)
        gt_boxes, gt_classes = _read_yolo_labels(label_dir / f"{image_path.stem}.txt", width, height)
        pred_boxes, pred_classes, pred_confidences = _prediction_arrays(result)
        matching = match_image(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_confidences, args.match_iou)
        image_records.append({
            "path": image_path,
            "width": width,
            "height": height,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
            "gt_size_bins": [size_bin(box, width, height) for box in gt_boxes],
            "gt_overlap_bins": overlap_bins(gt_boxes),
            "density_bin": density_bin(len(gt_boxes)),
            "pred_boxes": pred_boxes,
            "pred_classes": pred_classes,
            "pred_confidences": pred_confidences,
            "matching": matching,
        })
    prediction_seconds = time.perf_counter() - prediction_started
    missing = set(path.resolve() for path in image_paths) - seen_paths
    if missing:
        raise RuntimeError(f"Ultralytics returned predictions for {len(seen_paths)}/{len(image_paths)} images.")

    matrix, counts = _make_confusion_matrix(class_names, image_records)
    class_metrics: dict[str, dict[str, Any]] = {}
    for name in class_names:
        values = counts["class_counts"][name]
        class_metrics[name] = {**values, **_ratio_metrics(values)}
    total_counts = {key: sum(values[key] for values in counts["class_counts"].values()) for key in ("tp", "fp", "fn")}
    per_class_ap = _metrics_by_class(native_summary, class_names)
    instances_by_class = Counter(name for record in image_records for name in (class_names[int(class_id)] for class_id in record["gt_classes"]))
    for name in class_names:
        per_class_ap[name]["raw_annotation_instances"] = int(instances_by_class[name])
        if per_class_ap[name].get("instances") is None:
            per_class_ap[name]["instances"] = int(instances_by_class[name])
    native_ground_truth_objects = sum(int(values.get("instances") or 0) for values in per_class_ap.values())

    size_recall = _group_recall(image_records, "gt_size_bins", "gt_size_bins")
    density_recall = _group_recall(image_records, "density_bin")
    overlap_recall = _group_recall(image_records, "gt_overlap_bins", "gt_overlap_bins")
    error_rows = _error_rows(image_records, class_names)
    _write_csv(output_dir / "error_analysis.csv", error_rows)
    _plot_confusion(matrix, class_names, output_dir)
    _plot_class_metrics(per_class_ap, class_metrics, output_dir)
    _plot_group_recall(size_recall, "recall_by_size.png", "Recall by ground-truth object size", output_dir)
    _plot_group_recall(density_recall, "recall_by_density.png", "Recall by ground-truth scene density", output_dir)
    _plot_group_recall(overlap_recall, "recall_by_overlap.png", "Recall by ground-truth object overlap", output_dir)
    examples = _write_visual_examples(image_records, class_names, output_dir, args.max_examples)

    overall_metrics = {
        "map50_95": _metric_value(native_metrics, "metrics/mAP50-95(B)"),
        "map50": _metric_value(native_metrics, "metrics/mAP50(B)"),
    }
    operating = _ratio_metrics(total_counts)
    class_ap_values = [(name, values.get("ap50_95")) for name, values in per_class_ap.items() if values.get("ap50_95") is not None]
    best_class = max(class_ap_values, key=lambda item: item[1], default=(None, None))
    worst_class = min(class_ap_values, key=lambda item: item[1], default=(None, None))
    evaluator_speed = {
        key: json_safe(native_speed.get(key)) for key in ("preprocess", "inference", "postprocess")
    }
    inference_ms = evaluator_speed.get("inference")
    try:
        throughput = 1000.0 / float(inference_ms) if inference_ms and float(inference_ms) > 0 else None
    except (TypeError, ValueError):
        throughput = None
    model_name = str(config.get("model", checkpoint.parent.parent.name))
    summary = {
        "status": "completed",
        "model": model_name,
        "checkpoint": _relative_or_name(checkpoint),
        "split": args.split,
        "dataset": dataset_dir.name,
        "images_evaluated": len(image_records),
        "ground_truth_objects": sum(len(record["gt_classes"]) for record in image_records),
        "native_ground_truth_objects": native_ground_truth_objects,
        "label_integrity": {
            "exact_duplicate_label_rows": duplicate_labels,
            "source_annotations_modified": False,
        },
        "imgsz": imgsz,
        "batch": batch,
        "thresholds": {"confidence": args.conf, "match_iou": args.match_iou, "nms_iou": args.nms_iou, "localization_error_iou_min": LOCALIZATION_IOU},
        "global_metrics": overall_metrics,
        "operating_point_metrics": operating,
        "per_class_ap": per_class_ap,
        "per_class_operating_point": class_metrics,
        "best_class_by_ap50_95": {"class": best_class[0], "value": best_class[1]},
        "worst_class_by_ap50_95": {"class": worst_class[0], "value": worst_class[1]},
        "error_counts": counts["errors"],
        "class_confusions": counts["class_confusions"],
        "confusion_matrix_labels": [*class_names, "background"],
        "confusion_matrix_absolute": matrix.tolist(),
        "recall_by_size": size_recall,
        "recall_by_density": density_recall,
        "recall_by_overlap": overlap_recall,
        "examples": examples,
        "runtime": {
            "validation_wall_seconds": round(validation_seconds, 3),
            "error_analysis_pass_seconds": round(prediction_seconds, 3),
            "native_speed_ms_per_image": evaluator_speed,
            "native_inference_images_per_second": throughput,
        },
        "parameter_count": parameters,
        "hardware": _hardware_info(torch, args.device),
        "software": {"ultralytics": installed, "torch": str(torch.__version__)},
        "native_validation_directory": native_save_dir.name,
        "method": {
            "ap": "Ultralytics native validation; not recalculated on diagnostic subsets.",
            "matching": "Confidence-ordered greedy one-to-one diagnostic matching. An error can claim a ground-truth box before a later correct prediction.",
            "size_bins": {"small": "area <= 1% of image", "medium": "1% < area <= 10%", "large": "area > 10%"},
            "density_bins": {"sparse_1_5": "1-5 GT objects/image", "medium_6_9": "6-9 GT objects/image", "dense_10_plus": "10 or more GT objects/image"},
            "overlap_bins": {"low_lt_0_1": "max pairwise GT IoU < 0.1", "moderate_0_1_0_5": "0.1 <= max IoU < 0.5", "high_ge_0_5": "max IoU >= 0.5"},
            "bin_basis": "Predefined from docs/01_dataset_analysis.md and fixed independently of evaluation results.",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    _write_report(output_dir / "report.md", summary, output_dir, class_names)
    print(f"Evaluation artifacts saved to: {_relative_or_name(output_dir)}")
    return output_dir


def main() -> None:
    """CLI entry point."""
    evaluate(parse_args())


if __name__ == "__main__":
    main()
