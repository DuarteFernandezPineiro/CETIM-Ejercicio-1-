#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run validation-only NMS/confidence sensitivity and stratified error review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from evaluate_dataset import file_sha256
from evaluate_model import (
    IMAGE_EXTENSIONS,
    LOCALIZATION_IOU,
    _make_confusion_matrix,
    _prediction_arrays,
    _read_yolo_labels,
    _ratio_metrics,
    _write_csv,
    check_model_classes,
    density_bin,
    match_image,
    overlap_bins,
    predict_in_batches,
    size_bin,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "artifacts" / "yolo_dataset"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "detect" / "codd-yolov8n-img640-seed42" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "validation_followup"
PINNED_ULTRALYTICS_VERSION = "8.4.210"
NMS_IOUS = (0.50, 0.60, 0.70, 0.80, 0.90)
CONFIDENCE_THRESHOLDS = tuple([0.01, *[round(value, 2) for value in np.arange(0.05, 0.86, 0.05)]])
FOCUS_CLASSES = ("plastic", "pipes", "stone")
JOINT_KEYS = ("class", "size", "density", "overlap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25, help="Fixed confidence threshold for the NMS comparison")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42, help="Stable seed used only to sample visual review cases")
    parser.add_argument("--visuals-per-outcome", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz < 32 or args.batch < 1:
        raise SystemExit("imgsz must be >= 32 and batch must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.match_iou <= 1:
        raise SystemExit("conf and match-iou must be in [0, 1]")
    if args.visuals_per_outcome < 1:
        raise SystemExit("visuals-per-outcome must be positive")
    if args.device.lower() != "cpu" and not args.device.isdigit():
        raise SystemExit("device must be 'cpu' or a numeric CUDA device index")


def read_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"No class names in {data_yaml}")


def load_ground_truth(image_path: Path, label_dir: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
    boxes, classes = _read_yolo_labels(label_dir / f"{image_path.stem}.txt", width, height)
    return {
        "path": image_path,
        "width": width,
        "height": height,
        "gt_boxes": boxes,
        "gt_classes": classes,
        "gt_size_bins": [size_bin(box, width, height) for box in boxes],
        "gt_overlap_bins": overlap_bins(boxes),
        "density_bin": density_bin(len(boxes)),
        "density_count": len(boxes),
    }


def class_agnostic_matches(record: dict[str, Any], threshold: float) -> tuple[set[int], dict[int, int]]:
    """Greedy confidence-ordered one-to-one matches ignoring class labels."""
    gt_boxes = record["gt_boxes"]
    pred_boxes = record["pred_boxes"]
    pred_conf = record["pred_confidences"]
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return set(), {}
    from evaluate_model import box_iou_matrix

    ious = box_iou_matrix(pred_boxes, gt_boxes)
    unmatched = set(range(len(gt_boxes)))
    matched: dict[int, int] = {}
    for pred_index in np.argsort(-pred_conf, kind="stable"):
        if not unmatched:
            break
        gt_index = max(sorted(unmatched), key=lambda index: ious[pred_index, index])
        if ious[pred_index, gt_index] >= threshold:
            matched[gt_index] = int(pred_index)
            unmatched.remove(gt_index)
    return set(matched), matched


def duplicate_fp_count(record: dict[str, Any], match_iou: float) -> int:
    """Count unmatched FP boxes overlapping a GT already claimed by a higher-score box."""
    from evaluate_model import box_iou_matrix

    pred_boxes = record["pred_boxes"]
    gt_boxes = record["gt_boxes"]
    if not len(pred_boxes) or not len(gt_boxes):
        return 0
    ious = box_iou_matrix(pred_boxes, gt_boxes)
    confidence = record["pred_confidences"]
    matches = record["matching"]
    claimed = {int(match["gt_index"]): int(match["pred_index"]) for match in matches["matches"]}
    for error in matches["errors"]:
        if error["gt_index"] is not None and error["pred_index"] is not None:
            claimed[int(error["gt_index"])] = int(error["pred_index"])
    fp_indices = [error["pred_index"] for error in matches["errors"] if error["type"] == "false_positive"]
    duplicates = 0
    for pred_index in fp_indices:
        for gt_index, first_pred in claimed.items():
            if first_pred != pred_index and confidence[first_pred] > confidence[pred_index] and ious[pred_index, gt_index] >= match_iou:
                duplicates += 1
                break
    return duplicates


def attach_match(record: dict[str, Any], match_iou: float) -> dict[str, Any]:
    record["matching"] = match_image(
        record["gt_boxes"], record["gt_classes"], record["pred_boxes"],
        record["pred_classes"], record["pred_confidences"], match_iou,
    )
    localized_gt, geo_pairs = class_agnostic_matches(record, match_iou)
    record["localized_gt"] = localized_gt
    record["geo_pairs"] = geo_pairs
    record["duplicate_fp"] = duplicate_fp_count(record, match_iou)
    return record


def filtered_records(
    ground_truth: list[dict[str, Any]], predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    conf: float, match_iou: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for base in ground_truth:
        boxes, classes, scores = predictions[base["path"].name]
        keep = scores >= conf
        record = {**base, "pred_boxes": boxes[keep], "pred_classes": classes[keep], "pred_confidences": scores[keep]}
        records.append(attach_match(record, match_iou))
    return records


def save_prediction_cache(path: Path, predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], metadata: dict[str, Any]) -> None:
    names = list(predictions)
    offsets = [0]
    boxes, classes, scores = [], [], []
    for name in names:
        pred_boxes, pred_classes, pred_scores = predictions[name]
        boxes.append(np.asarray(pred_boxes, dtype=np.float32).reshape((-1, 4)))
        classes.append(np.asarray(pred_classes, dtype=np.int32).reshape((-1,)))
        scores.append(np.asarray(pred_scores, dtype=np.float32).reshape((-1,)))
        offsets.append(offsets[-1] + len(pred_scores))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        names=np.asarray(names, dtype=str),
        boxes=np.concatenate(boxes, axis=0) if boxes else np.empty((0, 4), dtype=np.float32),
        classes=np.concatenate(classes) if classes else np.empty((0,), dtype=np.int32),
        scores=np.concatenate(scores) if scores else np.empty((0,), dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
    )


def load_prediction_cache(path: Path, expected_metadata: dict[str, Any], expected_names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            names = data["names"].tolist()
            if metadata != expected_metadata or names != expected_names:
                return None
            boxes, classes, scores, offsets = data["boxes"], data["classes"], data["scores"], data["offsets"]
            return {
                name: (boxes[offsets[i]:offsets[i + 1]], classes[offsets[i]:offsets[i + 1]], scores[offsets[i]:offsets[i + 1]])
                for i, name in enumerate(names)
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def summarize_records(records: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    _, counts = _make_confusion_matrix(class_names, records)
    class_counts = counts["class_counts"]
    totals = {key: sum(values[key] for values in class_counts.values()) for key in ("tp", "fp", "fn")}
    standard = _ratio_metrics(totals)
    total_gt = sum(len(record["gt_classes"]) for record in records)
    localized = sum(len(record["localized_gt"]) for record in records)
    correctly_classified_localized = sum(
        1 for record in records for gt_index, pred_index in record["geo_pairs"].items()
        if record["gt_classes"][gt_index] == record["pred_classes"][pred_index]
    )
    errors = Counter(error["type"] for record in records for error in record["matching"]["errors"])
    duplicate_fp = sum(record["duplicate_fp"] for record in records)
    class_rows = {}
    for class_id, name in enumerate(class_names):
        class_records = [record for record in records if np.any(record["gt_classes"] == class_id)]
        gt_count = sum(int(np.sum(record["gt_classes"] == class_id)) for record in records)
        loc_count = sum(sum(1 for gt_index in record["localized_gt"] if record["gt_classes"][gt_index] == class_id) for record in records)
        good_count = sum(sum(1 for gt_index, pred_index in record["geo_pairs"].items() if record["gt_classes"][gt_index] == class_id and record["gt_classes"][gt_index] == record["pred_classes"][pred_index]) for record in records)
        vals = class_counts[name]
        class_rows[name] = {
            "ground_truth_objects": gt_count,
            "class_tp": vals["tp"], "class_fp": vals["fp"], "class_fn": vals["fn"],
            "class_precision": _ratio_metrics(vals)["precision"],
            "class_recall": _ratio_metrics(vals)["recall"],
            "class_f1": _ratio_metrics(vals)["f1"],
            "localized_any_class": loc_count,
            "localized_any_class_recall": loc_count / gt_count if gt_count else None,
            "correct_class_and_localized": good_count,
            "correct_class_and_localized_recall": good_count / gt_count if gt_count else None,
        }
    return {
        "images": len(records), "ground_truth_objects": total_gt,
        "predictions_after_conf": sum(len(record["pred_classes"]) for record in records),
        **standard,
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "localized_any_class": localized,
        "localized_any_class_recall": localized / total_gt if total_gt else None,
        "correct_class_and_localized": correctly_classified_localized,
        "correct_class_and_localized_recall": correctly_classified_localized / total_gt if total_gt else None,
        "unmatched_ground_truth_after_matching": errors["false_negative"],
        "classification_error_pairs": errors["classification_error"],
        "localization_error_pairs": errors["localization_error"],
        "unmatched_false_positive_predictions": errors["false_positive"],
        "duplicate_fp_subset": duplicate_fp,
        "class_metrics": class_rows,
    }


def subgroup_rows(records: list[dict[str, Any]], class_names: list[str]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    joint: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    marginals: dict[str, dict[tuple[str, str], list[int]]] = {
        factor: defaultdict(lambda: [0, 0, 0]) for factor in ("size", "density", "overlap")
    }
    for record in records:
        geo = record["geo_pairs"]
        for gt_index, class_id in enumerate(record["gt_classes"]):
            class_name = class_names[int(class_id)]
            factors = {
                "size": record["gt_size_bins"][gt_index],
                "density": record["density_bin"],
                "overlap": record["gt_overlap_bins"][gt_index],
            }
            key = (class_name, factors["size"], factors["density"], factors["overlap"])
            values = joint[key]
            values[0] += 1
            if gt_index in geo:
                values[1] += 1
                if record["gt_classes"][gt_index] == record["pred_classes"][geo[gt_index]]:
                    values[2] += 1
            for factor, group in factors.items():
                marginal = marginals[factor][(class_name, group)]
                marginal[0] += 1
                if gt_index in geo:
                    marginal[1] += 1
                    if record["gt_classes"][gt_index] == record["pred_classes"][geo[gt_index]]:
                        marginal[2] += 1
    joint_rows = []
    for (class_name, size, density, overlap), (n, localized, correct) in sorted(joint.items()):
        joint_rows.append({"class": class_name, "size": size, "density": density, "overlap": overlap,
                           "objects": n, "localized_any_class": localized, "correct_class_and_localized": correct,
                           "localization_recall": localized / n, "correct_class_recall": correct / n})
    marginal_rows: dict[str, list[dict[str, Any]]] = {}
    for factor, cells in marginals.items():
        marginal_rows[factor] = [
            {"class": class_name, "group": group, "objects": n, "localized_any_class": localized,
             "correct_class_and_localized": correct, "localization_recall": localized / n,
             "correct_class_recall": correct / n}
            for (class_name, group), (n, localized, correct) in sorted(cells.items())
        ]
    return joint_rows, marginal_rows


def plot_nms(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [row["nms_iou"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        axes[0].plot(xs, [row[key] for row in rows], marker="o", label=label)
    axes[0].set_title("Métricas globales (conf=0,25)")
    axes[0].set_xlabel("NMS IoU")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    for key, label in (("dense_recall", "Densa: ≥10 objetos"), ("moderate_overlap_recall", "Solapamiento 0,1–0,5"), ("high_overlap_recall", "Solapamiento ≥0,5")):
        axes[1].plot(xs, [row[key] for row in rows], marker="o", label=label)
    axes[1].set_title("Recall por escenario")
    axes[1].set_xlabel("NMS IoU")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    for key, label in (("unmatched_false_positive_predictions", "FP no emparejados"), ("duplicate_fp_subset", "Subconjunto duplicado"), ("unmatched_ground_truth_after_matching", "GT sin emparejar")):
        axes[2].plot(xs, [row[key] for row in rows], marker="o", label=label)
    axes[2].set_title("Conteos en validation")
    axes[2].set_xlabel("NMS IoU")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confidence(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = [row["confidence"] for row in rows]
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        axes[0].plot(xs, [row[key] for row in rows], marker="o", markersize=3, label=label)
    best = max(rows, key=lambda row: row["f1"] if row["f1"] is not None else -1)
    axes[0].axvline(best["confidence"], linestyle="--", color="gray", label=f"Máx. F1: {best['confidence']:.2f}")
    axes[0].set_title("Compromiso global, NMS IoU=0,70")
    axes[0].set_xlabel("Umbral de confianza")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].plot(xs, [row["dense_recall"] for row in rows], marker="o", markersize=3, label="Densa")
    axes[1].plot(xs, [row["moderate_overlap_recall"] for row in rows], marker="o", markersize=3, label="Overlap 0,1–0,5")
    axes[1].plot(xs, [row["high_overlap_recall"] for row in rows], marker="o", markersize=3, label="Overlap ≥0,5")
    axes[1].set_title("Recall en casos difíciles")
    axes[1].set_xlabel("Umbral de confianza")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_joint(marginals: dict[str, list[dict[str, Any]]], class_names: list[str], path: Path) -> None:
    factors = ("size", "density", "overlap")
    labels = {
        "size": (("small", "mediano", "grande"), ("small", "medium", "large")),
        "density": (("1–5", "6–9", "≥10"), ("sparse_1_5", "medium_6_9", "dense_10_plus")),
        "overlap": (("<0,1", "0,1–<0,5", "≥0,5"), ("low_lt_0_1", "moderate_0_1_0_5", "high_ge_0_5")),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    last_image = None
    for ax, factor in zip(axes, factors):
        display_labels, groups = labels[factor]
        values = np.full((len(class_names), len(groups)), np.nan)
        ns = np.zeros_like(values, dtype=int)
        lookup = {(row["class"], row["group"]): row for row in marginals[factor]}
        for yi, class_name in enumerate(class_names):
            for xi, group in enumerate(groups):
                row = lookup.get((class_name, group))
                if row:
                    values[yi, xi] = row["correct_class_recall"]
                    ns[yi, xi] = row["objects"]
        last_image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title({"size": "Clase × tamaño", "density": "Clase × densidad", "overlap": "Clase × solapamiento"}[factor])
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xticks(range(len(groups)), display_labels, fontsize=9)
        ax.set_xlabel("Grupo (n objetos; recall correcto)")
        for yi in range(len(class_names)):
            for xi in range(len(groups)):
                if ns[yi, xi]:
                    val = values[yi, xi]
                    ax.text(xi, yi, f"n={ns[yi, xi]}\n{val:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if val < 0.55 else "black")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, label="Recall de clase correcta", shrink=0.82, pad=0.03)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def deterministic_visuals(records: list[dict[str, Any]], class_names: list[str], args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    outcomes = ("correct", "false_negative", "classification_error", "localization_error")
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        matching = record["matching"]
        for match in matching["matches"]:
            gt_index, pred_index = match["gt_index"], match["pred_index"]
            class_name = class_names[int(record["gt_classes"][gt_index])]
            if class_name in FOCUS_CLASSES:
                candidates[(class_name, "correct")].append({"record": record, "gt_index": gt_index, "pred_index": pred_index, "iou": match["iou"]})
        for error in matching["errors"]:
            gt_index = error["gt_index"]
            if gt_index is None or error["type"] not in outcomes:
                continue
            class_name = class_names[int(record["gt_classes"][gt_index])]
            if class_name in FOCUS_CLASSES and error["type"] != "false_positive":
                candidates[(class_name, error["type"])].append({"record": record, "gt_index": gt_index, "pred_index": error["pred_index"], "iou": error["iou"]})

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    try:
        font = ImageFont.truetype(str(font_path), 20)
        small_font = ImageFont.truetype(str(font_path), 15)
    except OSError:
        font, small_font = ImageFont.load_default(), ImageFont.load_default()
    out_rows: list[dict[str, Any]] = []
    visual_root = output_dir / "visual_review"
    for class_name in FOCUS_CLASSES:
        for outcome in outcomes:
            pool = candidates[(class_name, outcome)]
            pool = sorted(pool, key=lambda item: hashlib.sha256(
                f"{args.seed}|{class_name}|{outcome}|{item['record']['path'].name}|{item['gt_index']}".encode()
            ).hexdigest())
            for sample_no, item in enumerate(pool[:args.visuals_per_outcome], 1):
                record = item["record"]
                gt_index = item["gt_index"]
                pred_index = item["pred_index"]
                image_path: Path = record["path"]
                with Image.open(image_path) as image:
                    original = image.convert("RGB")
                gt_panel, pred_panel = original.copy(), original.copy()
                gt_draw, pred_draw = ImageDraw.Draw(gt_panel), ImageDraw.Draw(pred_panel)
                for idx, (box, class_id) in enumerate(zip(record["gt_boxes"], record["gt_classes"])):
                    color, width = ((255, 220, 0), 5) if idx == gt_index else ((40, 190, 80), 2)
                    coordinates = tuple(int(v) for v in box)
                    gt_draw.rectangle(coordinates, outline=color, width=width)
                    label = f"GT {class_names[int(class_id)]}" + (" [OBJETIVO]" if idx == gt_index else "")
                    gt_draw.text((coordinates[0], max(0, coordinates[1] - 19)), label, fill=color, font=small_font, stroke_width=1, stroke_fill="black")
                for pidx, (box, class_id, score) in enumerate(zip(record["pred_boxes"], record["pred_classes"], record["pred_confidences"])):
                    color, width = ((255, 220, 0), 5) if pidx == pred_index else ((245, 65, 55), 2)
                    coordinates = tuple(int(v) for v in box)
                    pred_draw.rectangle(coordinates, outline=color, width=width)
                    label = f"{class_names[int(class_id)]} {score:.2f}" + (" [OBJETIVO]" if pidx == pred_index else "")
                    pred_draw.text((coordinates[0], max(0, coordinates[1] - 19)), label, fill=color, font=small_font, stroke_width=1, stroke_fill="black")
                width, height = original.size
                canvas = Image.new("RGB", (width * 2, height + 48), "white")
                canvas.paste(gt_panel, (0, 48))
                canvas.paste(pred_panel, (width, 48))
                draw = ImageDraw.Draw(canvas)
                title = f"{class_name} | {outcome} | GT #{gt_index} | IoU={item['iou']:.3f} | {image_path.name}"
                draw.text((12, 10), title, fill="black", font=font)
                draw.text((width // 2 - 45, 25), "Ground truth", fill="black", font=small_font)
                draw.text((width + width // 2 - 35, 25), "Predicciones", fill="black", font=small_font)
                relative = Path("visual_review") / class_name / f"{outcome}_{sample_no:02d}_{image_path.stem}.jpg"
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                canvas.save(destination, quality=90)
                out_rows.append({
                    "class": class_name, "outcome": outcome, "image": image_path.name,
                    "visual": relative.as_posix(), "gt_object_index_in_label_file": gt_index,
                    "predicted_class": class_names[int(record["pred_classes"][pred_index])] if pred_index is not None else "",
                    "confidence": float(record["pred_confidences"][pred_index]) if pred_index is not None else "",
                    "matched_iou": float(item["iou"]), "size": record["gt_size_bins"][gt_index],
                    "scene_objects": record["density_count"], "density": record["density_bin"],
                    "overlap": record["gt_overlap_bins"][gt_index],
                    "review_possible_cause": "", "review_annotation_consistency": "", "review_notes": "",
                })
    return out_rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    data_yaml = dataset_dir / "data.yaml"
    validation_report = dataset_dir / "validation_report.json"
    if not data_yaml.is_file() or not validation_report.is_file():
        raise SystemExit("Validated YOLO dataset is required (data.yaml and validation_report.json).")
    if json.loads(validation_report.read_text(encoding="utf-8")).get("passed") is not True:
        raise SystemExit("Converted dataset validation did not pass.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    val_images = sorted(
        path for path in (dataset_dir / "images" / "val").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels_dir = dataset_dir / "labels" / "val"
    if not val_images:
        raise SystemExit("No validation images found.")
    class_names = read_names(data_yaml)
    if not set(FOCUS_CLASSES).issubset(class_names):
        raise SystemExit("Expected focus classes plastic, pipes, and stone are not all present.")

    from importlib.metadata import version
    from ultralytics import YOLO

    installed_version = version("ultralytics")
    if installed_version != PINNED_ULTRALYTICS_VERSION:
        raise SystemExit(f"Expected ultralytics=={PINNED_ULTRALYTICS_VERSION}, found {installed_version}.")
    model = YOLO(str(checkpoint))
    check_model_classes(model, class_names)
    low_conf = min(CONFIDENCE_THRESHOLDS)
    ground_truth = [load_ground_truth(path, labels_dir) for path in val_images]
    predictions_by_iou: dict[float, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    timings: dict[str, float] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / ".prediction_cache"
    image_names = [path.name for path in val_images]
    cache_settings = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "image_sha256": [file_sha256(path) for path in val_images],
        "conf_floor": low_conf,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "ultralytics": installed_version,
        "torch": version("torch"),
    }

    for nms_iou in NMS_IOUS:
        cache_path = cache_dir / f"nms_{nms_iou:.2f}.npz"
        metadata = {**cache_settings, "nms_iou": nms_iou}
        prediction_map = load_prediction_cache(cache_path, metadata, image_names)
        if prediction_map is not None:
            print(f"Usando caché de validation: NMS IoU={nms_iou:.2f}.", flush=True)
            timings[f"nms_{nms_iou:.2f}_seconds"] = 0.0
        else:
            print(f"Prediciendo validation: NMS IoU={nms_iou:.2f}, conf mínima={low_conf:.2f} ...", flush=True)
            started = time.perf_counter()
            prediction_map = {}
            stream = predict_in_batches(
                model, val_images,
                imgsz=args.imgsz, batch=args.batch, conf=low_conf, iou=nms_iou,
                device=args.device, max_det=300, verbose=False, save=False,
            )
            for result in stream:
                path = Path(result.path)
                prediction_map[path.name] = _prediction_arrays(result)
            if len(prediction_map) != len(val_images):
                raise RuntimeError(f"Expected {len(val_images)} predictions; received {len(prediction_map)}.")
            save_prediction_cache(cache_path, prediction_map, metadata)
            timings[f"nms_{nms_iou:.2f}_seconds"] = time.perf_counter() - started
        predictions_by_iou[nms_iou] = prediction_map

    nms_rows: list[dict[str, Any]] = []
    records_by_iou: dict[float, list[dict[str, Any]]] = {}
    for nms_iou in NMS_IOUS:
        records = filtered_records(ground_truth, predictions_by_iou[nms_iou], args.conf, args.match_iou)
        records_by_iou[nms_iou] = records
        summary = summarize_records(records, class_names)
        for density_key in ("sparse_1_5", "medium_6_9", "dense_10_plus"):
            subset = [record for record in records if record["density_bin"] == density_key]
            subtotal = summarize_records(subset, class_names) if subset else {}
            summary[f"{density_key}_objects"] = subtotal.get("ground_truth_objects", 0)
            summary[f"{density_key}_localization_recall"] = subtotal.get("localized_any_class_recall")
            summary[f"{density_key}_correct_class_recall"] = subtotal.get("correct_class_and_localized_recall")
        summary["dense_recall"] = summary["dense_10_plus_correct_class_recall"]
        for overlap_key in ("low_lt_0_1", "moderate_0_1_0_5", "high_ge_0_5"):
            subset = [record for record in records if any(group == overlap_key for group in record["gt_overlap_bins"])]
            obj_count = loc_count = class_count = 0
            for record in subset:
                for gt_index, group in enumerate(record["gt_overlap_bins"]):
                    if group != overlap_key:
                        continue
                    obj_count += 1
                    if gt_index in record["localized_gt"]:
                        loc_count += 1
                        pred_index = record["geo_pairs"][gt_index]
                        if record["gt_classes"][gt_index] == record["pred_classes"][pred_index]:
                            class_count += 1
            summary[f"{overlap_key}_objects"] = obj_count
            summary[f"{overlap_key}_localization_recall"] = loc_count / obj_count if obj_count else None
            summary[f"{overlap_key}_correct_class_recall"] = class_count / obj_count if obj_count else None
        summary["moderate_overlap_recall"] = summary["moderate_0_1_0_5_correct_class_recall"]
        summary["high_overlap_recall"] = summary["high_ge_0_5_correct_class_recall"]
        summary["nms_iou"] = nms_iou
        nms_rows.append(summary)

    confidence_rows: list[dict[str, Any]] = []
    conf_predictions = predictions_by_iou[0.70]
    for conf in CONFIDENCE_THRESHOLDS:
        records = filtered_records(ground_truth, conf_predictions, conf, args.match_iou)
        row = {"confidence": conf, **summarize_records(records, class_names)}
        for key in ("dense_10_plus", "moderate_0_1_0_5", "high_ge_0_5"):
            if key == "dense_10_plus":
                subset = [record for record in records if record["density_bin"] == key]
                row["dense_objects"] = sum(len(record["gt_classes"]) for record in subset)
                row["dense_recall"] = summarize_records(subset, class_names)["correct_class_and_localized_recall"] if subset else None
            else:
                count = hits = 0
                for record in records:
                    for gt_index, group in enumerate(record["gt_overlap_bins"]):
                        if group == key:
                            count += 1
                            if gt_index in record["geo_pairs"] and record["gt_classes"][gt_index] == record["pred_classes"][record["geo_pairs"][gt_index]]:
                                hits += 1
                row[{"moderate_0_1_0_5": "moderate_overlap_recall", "high_ge_0_5": "high_overlap_recall"}[key]] = hits / count if count else None
        confidence_rows.append(row)

    baseline_records = records_by_iou[0.70]
    joint_rows, marginal_rows = subgroup_rows(baseline_records, class_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "nms_sensitivity.csv", nms_rows)
    _write_csv(args.output_dir / "confidence_sensitivity.csv", confidence_rows)
    _write_csv(args.output_dir / "joint_class_size_density_overlap.csv", joint_rows)
    for factor, rows in marginal_rows.items():
        _write_csv(args.output_dir / f"class_by_{factor}.csv", rows)
    plot_nms(nms_rows, args.output_dir / "nms_sensitivity.png")
    plot_confidence(confidence_rows, args.output_dir / "confidence_sensitivity.png")
    plot_joint(marginal_rows, class_names, args.output_dir / "class_factor_heatmaps.png")
    visual_rows = deterministic_visuals(baseline_records, class_names, args, args.output_dir)
    _write_csv(args.output_dir / "visual_review.csv", visual_rows)

    best_f1 = max(confidence_rows, key=lambda row: row["f1"] if row["f1"] is not None else -1)
    high_recall_options = [row for row in confidence_rows if row["recall"] is not None and row["recall"] >= 0.85]
    high_precision_options = [row for row in confidence_rows if row["precision"] is not None and row["precision"] >= 0.90]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "val", "test_split_used": False,
        "model": "yolov8n", "checkpoint": "runs/detect/codd-yolov8n-img640-seed42/weights/best.pt",
        "ultralytics": installed_version, "imgsz": args.imgsz, "batch": args.batch,
        "fixed_nms_confidence": args.conf, "match_iou": args.match_iou,
        "nms_confidence_floor": low_conf, "nms_iou_values": list(NMS_IOUS),
        "confidence_values": list(CONFIDENCE_THRESHOLDS), "images": len(val_images),
        "ground_truth_objects": sum(len(item["gt_classes"]) for item in ground_truth),
        "nms_results": [{key: value for key, value in row.items() if key != "class_metrics"} for row in nms_rows],
        "confidence_results": [{key: value for key, value in row.items() if key != "class_metrics"} for row in confidence_rows],
        "class_metrics_at_baseline": nms_rows[NMS_IOUS.index(0.70)]["class_metrics"],
        "confidence_candidate_points": {
            "maximum_f1": {key: best_f1[key] for key in ("confidence", "precision", "recall", "f1")},
            "highest_recall_with_precision_at_least_0_90": {key: max(high_precision_options, key=lambda row: row["recall"])[key] for key in ("confidence", "precision", "recall", "f1")} if high_precision_options else None,
            "highest_precision_with_recall_at_least_0_85": {key: max(high_recall_options, key=lambda row: row["precision"])[key] for key in ("confidence", "precision", "recall", "f1")} if high_recall_options else None,
        },
        "joint_strata_csv": "joint_class_size_density_overlap.csv",
        "visual_selection": {"seed": args.seed, "per_class_outcome": args.visuals_per_outcome, "outcomes": ["correct", "false_negative", "classification_error", "localization_error"], "csv": "visual_review.csv"},
        "runtime_seconds_by_nms": timings,
        "method": {
            "predictions": "Ultralytics model.predict on validation only; NMS IoU varied, all other inference arguments fixed.",
            "confidence_sweep": "Filter NMS=0.70 detections generated at the minimum confidence floor; higher confidence filtering is equivalent to the same NMS ordering.",
            "localization_recall": "Confidence-ordered greedy one-to-one GT/prediction matching at match IoU, class ignored.",
            "correct_class_recall": "The same geometric assignment, requiring predicted and GT classes to agree.",
            "duplicate_fp_subset": "Unmatched FP prediction overlapping a GT at or above match IoU already claimed by a higher-confidence prediction.",
            "joint_strata": "Each GT instance counted by class, size, scene density and max GT overlap; subgroup results are descriptive, not causal.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for cached_file in cache_dir.glob("*.npz"):
        cached_file.unlink()
    cache_dir.rmdir()
    print(f"Resultados guardados en {args.output_dir}")
    print(f"Ejemplos visuales disponibles: {len(visual_rows)}; revisar visual_review.csv")
    print(f"Mejor F1 en la rejilla de confianza: {best_f1['f1']:.4f} a conf={best_f1['confidence']:.2f}")


if __name__ == "__main__":
    main()
