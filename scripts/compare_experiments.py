#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate and compare saved Ultralytics training experiments using validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = PROJECT_ROOT / "runs" / "detect"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "experiment_comparison"
SUMMARY_FILENAME = "experiment_summary.json"
RESULTS_FILENAME = "results.csv"
ARGS_FILENAME = "args.yaml"
COMPARABILITY_FIELDS = (
    "dataset", "imgsz", "batch", "seed", "optimizer", "lr0", "weight_decay",
    "amp", "epochs_max", "ultralytics", "torch", "hardware",
)


@dataclass
class Experiment:
    """Normalized experiment record used for CSV, JSON, report, and plots."""

    run_dir: Path
    status: str
    model: str | None = None
    family: str | None = None
    size: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    smoke_test: bool = False
    record: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON while returning None for absent or malformed files."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, returning an empty mapping for absent or invalid YAML."""
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read Ultralytics' epoch log without coercing missing values to zero."""
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    except (OSError, csv.Error):
        return []


def _portable_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-specific absolute paths from metadata copied into artifacts."""
    portable = dict(values)
    for key, value in portable.items():
        if not isinstance(value, str):
            continue
        candidate = Path(value)
        if candidate.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
            portable[key] = candidate.name
    return portable


def _number(value: Any) -> float | None:
    """Parse a finite numeric value, preserving unavailable metrics as None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def calculate_f1(precision: float | None, recall: float | None) -> float | None:
    """Calculate F1 only when both precision and recall exist."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_epoch_from_results(
    rows: list[dict[str, str]], metric_key: str = "metrics/mAP50-95(B)"
) -> tuple[int | None, float | None]:
    """Find the 1-based epoch maximizing the checkpoint criterion in results.csv."""
    first_epoch = next((_number(row.get("epoch")) for row in rows if _number(row.get("epoch")) is not None), None)
    zero_based = first_epoch is not None and first_epoch == 0
    candidates: list[tuple[int, float, int]] = []
    for index, row in enumerate(rows):
        score = _number(row.get(metric_key))
        epoch_value = _number(row.get("epoch"))
        if score is None:
            continue
        if epoch_value is None:
            epoch = index + 1
        else:
            epoch = int(epoch_value) + 1 if zero_based else int(epoch_value)
        candidates.append((epoch, score, index))
    if not candidates:
        return None, None
    # max() returns the first row among score ties because candidates retain CSV order.
    epoch, score, _ = max(candidates, key=lambda item: item[1])
    return epoch, score


def _model_identity(summary: dict[str, Any], args: dict[str, Any], run_name: str) -> tuple[str | None, str | None, str | None]:
    """Extract architecture family and size from explicit metadata or the run name."""
    model = str(summary.get("model") or args.get("model") or "")
    if model.endswith(".pt"):
        model = model[:-3]
    if not model:
        match = re.search(r"(yolov8|yolo11|yolo26)([nsmlx])", run_name.lower())
        model = "".join(match.groups()) if match else ""
    match = re.search(r"(yolov8|yolo11|yolo26)([nsmlx])", model.lower())
    if not match:
        match = re.search(r"(yolov8|yolo11|yolo26)([nsmlx])", run_name.lower())
    if not match:
        return model or None, None, None
    family, size = match.groups()
    return f"{family}{size}", family, size


def _as_float(value: Any) -> float | None:
    return _number(value)


def _as_int(value: Any) -> int | None:
    """Parse a finite integer-like value without converting missing values to zero."""
    number = _number(value)
    return int(number) if number is not None else None


def _summary_metric(summary: dict[str, Any], key: str) -> float | None:
    return _number((summary.get("validation_metrics") or {}).get(key))


def _derive_record(experiment: Experiment) -> dict[str, Any]:
    """Combine summary, args.yaml, and results.csv without fabricating values."""
    summary = experiment.summary
    training = summary.get("training") or {}
    configuration = summary.get("configuration") or {}
    args = experiment.args
    model, family, size = _model_identity(summary, {**args, **configuration}, experiment.run_dir.name)
    experiment.model, experiment.family, experiment.size = model, family, size

    get = lambda key, default=None: configuration.get(key, training.get(key, args.get(key, default)))
    model_index = summary.get("validation_per_class") or []
    class_ap: dict[str, Any] = {}
    for row in model_index:
        if not isinstance(row, dict):
            continue
        name = row.get("Class", row.get("class"))
        if name:
            class_ap[str(name)] = {
                "ap50": _as_float(row.get("mAP50", row.get("AP50"))),
                "ap50_95": _as_float(row.get("mAP50-95", row.get("AP50-95"))),
                "precision": _as_float(row.get("Box-P", row.get("precision"))),
                "recall": _as_float(row.get("Box-R", row.get("recall"))),
                "f1": _as_float(row.get("Box-F1", row.get("f1"))),
                "instances": row.get("Instances", row.get("instances")),
            }

    selection_metric = str(training.get("best_epoch_selection_metric") or "metrics/mAP50-95(B)")
    if experiment.rows and selection_metric not in experiment.rows[0]:
        experiment.warnings.append(f"checkpoint selection metric '{selection_metric}' is absent from results.csv")
        selection_metric = "metrics/mAP50-95(B)"
    csv_best_epoch, csv_best_map = best_epoch_from_results(experiment.rows, selection_metric)
    summary_best_epoch = training.get("best_epoch")
    best_epoch = csv_best_epoch if csv_best_epoch is not None else summary_best_epoch
    if summary_best_epoch is not None and csv_best_epoch is not None and int(summary_best_epoch) != csv_best_epoch:
        experiment.warnings.append(
            f"best epoch mismatch: summary says {summary_best_epoch}, results.csv selects epoch {csv_best_epoch} by mAP50-95"
        )
    summary_best_score = _number(training.get("best_epoch_metric_value"))
    if summary_best_score is not None and csv_best_map is not None and not math.isclose(summary_best_score, csv_best_map, rel_tol=0, abs_tol=1e-5):
        experiment.warnings.append("summary best-epoch metric differs from results.csv best mAP50-95")
    if experiment.rows and training.get("epochs_executed") is not None and int(training["epochs_executed"]) != len(experiment.rows):
        experiment.warnings.append(f"epoch-count mismatch: summary says {training['epochs_executed']}, results.csv has {len(experiment.rows)} rows")

    precision = _summary_metric(summary, "metrics/precision(B)")
    recall = _summary_metric(summary, "metrics/recall(B)")
    map50 = _summary_metric(summary, "metrics/mAP50(B)")
    map50_95 = _summary_metric(summary, "metrics/mAP50-95(B)")
    if map50_95 is None and csv_best_map is not None:
        map50_95 = csv_best_map
    if map50 is None and experiment.rows:
        best_row = max(experiment.rows, key=lambda row: _number(row.get("metrics/mAP50-95(B)")) if _number(row.get("metrics/mAP50-95(B)")) is not None else float("-inf"))
        map50 = _number(best_row.get("metrics/mAP50(B)"))
        precision = precision if precision is not None else _number(best_row.get("metrics/precision(B)"))
        recall = recall if recall is not None else _number(best_row.get("metrics/recall(B)"))
    explicit_f1 = _summary_metric(summary, "metrics/F1(B)")
    f1 = explicit_f1 if explicit_f1 is not None else calculate_f1(precision, recall)

    configured_epochs = get("epochs", training.get("epochs_max"))
    configured_epochs = _as_int(configured_epochs)
    smoke_name = bool(re.search(r"(^|[-_])(smoke|dryrun|sanity)([-_]|$)", experiment.run_dir.name.lower()))
    experiment.smoke_test = smoke_name or (configured_epochs is not None and configured_epochs <= 3)

    per_class_ap_summary = class_ap or summary.get("validation_per_class_ap", {}) or {}
    hardware = summary.get("hardware") or {}
    software = summary.get("software") or {}
    run_directory = str(experiment.run_dir)
    try:
        run_directory = experiment.run_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        run_directory = experiment.run_dir.name
    declared_splits = (
        training.get("validation_split"), configuration.get("validation_split"), args.get("split")
    )
    validation_split = next((split for split in declared_splits if split is not None), None)
    test_split_used = training.get("test_split_used", configuration.get("test_split_used"))
    if "test" in declared_splits or configuration.get("test_split_used") is True:
        test_split_used = True
    outside_protocol = test_split_used is True or any(split not in (None, "val") for split in declared_splits)
    if outside_protocol:
        experiment.warnings.append("Excluded from comparison: the run declares a non-validation split or use of test")
    record = {
        "run_directory": run_directory,
        "run_name": experiment.run_dir.name,
        "status": experiment.status,
        "eligible_for_ranking": experiment.status == "completed" and map50_95 is not None and not outside_protocol,
        "smoke_test": experiment.smoke_test,
        "model": model,
        "family": family,
        "size": size,
        "dataset": Path(str(summary.get("dataset") or args.get("data"))).name if (summary.get("dataset") or args.get("data")) else None,
        "imgsz": _as_float(get("imgsz")),
        "seed": _as_float(get("seed")),
        "batch": _as_float(training.get("batch_requested", get("batch"))),
        "optimizer": get("optimizer"),
        "lr0": _as_float(get("lr0")),
        "weight_decay": _as_float(get("weight_decay")),
        "amp": get("amp"),
        "epochs_max": configured_epochs,
        "epochs_executed": _as_int(training.get("epochs_executed", len(experiment.rows) or None)),
        "best_epoch": _as_int(best_epoch),
        "best_epoch_summary": _as_int(summary_best_epoch),
        "best_epoch_csv": _as_int(csv_best_epoch),
        "best_epoch_consistent": None if summary_best_epoch is None or csv_best_epoch is None else int(summary_best_epoch) == csv_best_epoch,
        "best_epoch_metric_value": csv_best_map if csv_best_map is not None else summary_best_score,
        "best_epoch_selection_metric": selection_metric,
        "map50_95": map50_95,
        "map50": map50,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_ap": per_class_ap_summary,
        "training_seconds": _as_float(training.get("training_seconds")),
        "validation_seconds": _as_float(training.get("validation_seconds")),
        "inference_latency_ms_per_image": _as_float(summary.get("inference_latency_ms_per_image")),
        "inference_images_per_second": _as_float(summary.get("inference_images_per_second")),
        "hardware": hardware,
        "software": software,
        "warnings": experiment.warnings,
        "configuration": _portable_mapping(configuration),
        "validation_split": validation_split,
        "test_split_used": test_split_used,
    }
    experiment.record = record
    return record


def load_experiment(run_dir: Path) -> Experiment:
    """Load one run from experiment_summary.json, results.csv, and args.yaml."""
    summary_path = run_dir / SUMMARY_FILENAME
    results_path = run_dir / RESULTS_FILENAME
    args_path = run_dir / ARGS_FILENAME
    summary = _read_json(summary_path) or {}
    args = _read_yaml(args_path)
    rows = _read_csv(results_path)
    warnings: list[str] = []
    if not summary_path.is_file():
        warnings.append("experiment_summary.json is missing")
    elif not summary:
        warnings.append("experiment_summary.json is invalid or unreadable")
    if not results_path.is_file():
        warnings.append("results.csv is missing")
    elif not rows:
        warnings.append("results.csv is empty or unreadable")
    if not args_path.is_file():
        warnings.append("args.yaml is missing")
    status = str(summary.get("status", "incomplete"))
    if status not in {"completed", "failed", "incomplete"}:
        status = "incomplete"
    if status == "completed" and not summary.get("validation_metrics"):
        warnings.append("completed summary has no validation metrics")
        status = "incomplete"
    experiment = Experiment(run_dir=run_dir, status=status, summary=summary, args=args, rows=rows, warnings=warnings)
    _derive_record(experiment)
    if status == "completed":
        best_path = run_dir / "weights" / "best.pt"
        if not best_path.is_file():
            experiment.warnings.append("best.pt checkpoint is missing")
    return experiment


def discover_experiments(runs_dir: Path) -> list[Experiment]:
    """Discover run directories from the three training artifacts, excluding validation output folders."""
    if not runs_dir.is_dir():
        return []
    candidates: set[Path] = set()
    for filename in (SUMMARY_FILENAME, RESULTS_FILENAME, ARGS_FILENAME):
        for path in runs_dir.rglob(filename):
            relative_parts = path.relative_to(runs_dir).parts
            if any(part.lower() in {"validation", "val", "test", "predict"} for part in relative_parts[:-1]):
                continue
            candidates.add(path.parent)
    return [load_experiment(path) for path in sorted(candidates)]


def _comparison_warnings(records: list[dict[str, Any]]) -> list[str]:
    """Warn about metadata differences that weaken a direct ranking."""
    complete = [record for record in records if record["eligible_for_ranking"] and not record["smoke_test"]]
    warnings: list[str] = []
    if len(complete) < 2:
        return warnings
    for field_name in COMPARABILITY_FIELDS:
        values: dict[str, list[str]] = defaultdict(list)
        for record in complete:
            if field_name == "hardware":
                value = json.dumps(record.get("hardware"), sort_keys=True, ensure_ascii=False)
            elif field_name == "ultralytics" or field_name == "torch":
                value = (record.get("software") or {}).get(field_name)
            else:
                value = record.get(field_name)
            if value is not None:
                values[str(value)].append(record["run_name"])
        if len(values) > 1:
            detail = "; ".join(f"{value}: {', '.join(names)}" for value, names in values.items())
            warnings.append(f"Different {field_name} values may limit fairness: {detail}.")
    return warnings


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten comparison fields while encoding per-class AP as JSON."""
    fields = (
        "run_name", "status", "eligible_for_ranking", "smoke_test", "model", "family", "size", "dataset",
        "imgsz", "seed", "batch", "optimizer", "lr0", "weight_decay", "amp", "epochs_max",
        "epochs_executed", "best_epoch", "best_epoch_summary", "best_epoch_csv", "best_epoch_consistent", "best_epoch_selection_metric",
        "best_epoch_metric_value", "map50_95", "map50", "precision", "recall", "f1", "training_seconds",
        "validation_seconds", "inference_latency_ms_per_image", "inference_images_per_second",
        "validation_split", "test_split_used",
    )
    row = {field: record.get(field) for field in fields}
    row["per_class_ap_json"] = json.dumps(record.get("per_class_ap", {}), ensure_ascii=False, sort_keys=True)
    row["hardware_json"] = json.dumps(record.get("hardware", {}), ensure_ascii=False, sort_keys=True)
    row["software_json"] = json.dumps(record.get("software", {}), ensure_ascii=False, sort_keys=True)
    row["warnings"] = " | ".join(record.get("warnings", []))
    row["run_directory"] = record.get("run_directory")
    return row


def _save_comparison_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one row per discovered run, with blank cells for absent values."""
    rows = [_csv_row(record) for record in records]
    if not rows:
        rows = [{"run_name": "", "status": "no_experiments_found"}]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_global(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot validation mAP, precision, recall, and derived F1 for eligible experiments."""
    if not records:
        return
    labels = [record["model"] or record["run_name"] for record in records]
    metric_keys = (("map50_95", "mAP50-95"), ("map50", "mAP50"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
    positions = np.arange(len(labels))
    width = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.8), 5))
    for index, (key, label) in enumerate(metric_keys):
        values = [record.get(key) for record in records]
        ax.bar(positions + (index - (len(metric_keys) - 1) / 2) * width,
               [value if value is not None else np.nan for value in values], width=width, label=label)
    ax.set_xticks(positions, labels=labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.set_title("Validation metrics by experiment")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "global_metrics.png", dpi=160)
    plt.close(fig)


def _read_curve_column(rows: list[dict[str, str]], key: str) -> tuple[list[float], list[float]]:
    """Return 1-based epoch and finite values for a results.csv metric column."""
    first_epoch = next((_number(row.get("epoch")) for row in rows if _number(row.get("epoch")) is not None), None)
    zero_based = first_epoch is not None and first_epoch == 0
    epochs: list[float] = []
    values: list[float] = []
    for index, row in enumerate(rows):
        value = _number(row.get(key))
        if value is None:
            continue
        epoch_value = _number(row.get("epoch"))
        epoch = int(epoch_value) + 1 if epoch_value is not None and zero_based else int(epoch_value) if epoch_value is not None else index + 1
        epochs.append(float(epoch))
        values.append(value)
    return epochs, values


def _plot_learning_curves(experiments: list[Experiment], output_dir: Path) -> None:
    """Plot validation mAP and training/validation losses from epoch logs."""
    labeled = [experiment for experiment in experiments if experiment.rows]
    if not labeled:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for experiment in labeled:
        epochs, values = _read_curve_column(experiment.rows, "metrics/mAP50-95(B)")
        if values:
            ax.plot(epochs, values, label=experiment.model or experiment.run_dir.name)
    ax.set_xlabel("Epoch (from results.csv)")
    ax.set_ylabel("Validation mAP50-95")
    ax.set_ylim(0, 1)
    ax.set_title("Validation learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    loss_keys = (("train/box_loss", "Train box"), ("train/cls_loss", "Train class"), ("train/dfl_loss", "Train DFL"),
                 ("val/box_loss", "Val box"), ("val/cls_loss", "Val class"), ("val/dfl_loss", "Val DFL"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title) in zip(axes.flat, loss_keys):
        for experiment in labeled:
            epochs, values = _read_curve_column(experiment.rows, key)
            if values:
                ax.plot(epochs, values, label=experiment.model or experiment.run_dir.name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("Training and validation loss curves", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _class_values(record: dict[str, Any]) -> dict[str, Any]:
    """Return normalized class AP values from the experiment record."""
    result = record.get("per_class_ap") or {}
    return result if isinstance(result, dict) else {}


def _plot_class_ap(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Write grouped per-class AP chart and model-by-class heatmap."""
    labels = sorted({name for record in records for name in _class_values(record)})
    if not records or not labels:
        return
    values = np.asarray([[_number(_class_values(record).get(name, {}).get("ap50_95")) for name in labels] for record in records], dtype=object)
    numeric = np.asarray([[np.nan if value is None else float(value) for value in row] for row in values], dtype=float)
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 1.1), 5))
    positions = np.arange(len(labels))
    width = 0.8 / max(1, len(records))
    for index, record in enumerate(records):
        offset = (index - (len(records) - 1) / 2) * width
        ax.bar(positions + offset, numeric[index], width=width, label=record.get("model") or record["run_name"])
    ax.set_xticks(positions, labels=labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP50-95")
    ax.set_title("Validation AP50-95 by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_ap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), max(3.5, len(records) * 0.7)))
    image = ax.imshow(numeric, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax, label="AP50-95")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right")
    ax.set_yticks(range(len(records)), labels=[record.get("model") or record["run_name"] for record in records])
    for row in range(numeric.shape[0]):
        for column in range(numeric.shape[1]):
            if np.isfinite(numeric[row, column]):
                ax.text(column, row, f"{numeric[row, column]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if numeric[row, column] < 0.55 else "black")
    ax.set_title("Class/model AP50-95 heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "class_model_heatmap.png", dpi=160)
    plt.close(fig)


def _plot_training_time(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Compare recorded training duration for eligible experiments."""
    available = [record for record in records if record.get("training_seconds") is not None]
    if not available:
        return
    labels = [record.get("model") or record["run_name"] for record in available]
    hours = [float(record["training_seconds"]) / 3600 for record in available]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4.5))
    ax.bar(labels, hours, color="#517ca3")
    ax.set_ylabel("Training hours")
    ax.set_title("Recorded training time")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_time.png", dpi=160)
    plt.close(fig)


def _plot_accuracy_latency(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Show the validation accuracy/latency trade-off."""
    available = [record for record in records if record.get("map50_95") is not None and record.get("inference_latency_ms_per_image") is not None]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for record in available:
        x = float(record["inference_latency_ms_per_image"])
        y = float(record["map50_95"])
        ax.scatter(x, y, s=55)
        ax.annotate(record.get("model") or record["run_name"], (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Inference latency (ms/image, validation measurement)")
    ax.set_ylabel("Validation mAP50-95")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy and inference latency")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "map_latency.png", dpi=160)
    plt.close(fig)


def _format_number(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _write_report(path: Path, records: list[dict[str, Any]], warnings: list[str], failed_models: list[str], include_smoke: bool) -> None:
    """Write a ranking and methodological warnings without treating missing values as zero."""
    eligible = [record for record in records if record["eligible_for_ranking"] and (include_smoke or not record["smoke_test"])]
    eligible.sort(key=lambda record: (record.get("map50_95") is not None, record.get("map50_95") or float("-inf")), reverse=True)
    lines = [
        "# YOLO experiment comparison",
        "",
        "Only validation metrics are used for selection. The reserved test split is never loaded by this comparator.",
        "",
        "## Ranking by validation mAP50-95",
        "",
        "| Rank | Model | mAP50-95 | mAP50 | Precision | Recall | F1 | Epochs | Best epoch (CSV) | Train h | Latency ms/img |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not eligible:
        lines.append("| — | No completed, eligible experiments found | — | — | — | — | — | — | — | — | — |")
    for index, record in enumerate(eligible, start=1):
        train_hours = _number(record.get("training_seconds"))
        train_hours_text = f"{train_hours / 3600:.2f}" if train_hours is not None else "—"
        lines.append(
            f"| {index} | {record.get('model') or record['run_name']} | {_format_number(record.get('map50_95'))} | "
            f"{_format_number(record.get('map50'))} | {_format_number(record.get('precision'))} | {_format_number(record.get('recall'))} | "
            f"{_format_number(record.get('f1'))} | {record.get('epochs_executed') or '—'} | {record.get('best_epoch_csv') or '—'} | "
            f"{train_hours_text} | {_format_number(record.get('inference_latency_ms_per_image'), 3)} |"
        )
    lines.extend(["", "## Per-class AP50-95", ""])
    classes = sorted({class_name for record in eligible for class_name in _class_values(record)})
    if classes and eligible:
        lines.append("| Class | " + " | ".join(str(record.get("model") or record["run_name"]) for record in eligible) + " |")
        lines.append("|---|" + "---:|" * len(eligible))
        for class_name in classes:
            row = [class_name]
            for record in eligible:
                value = _class_values(record).get(class_name, {}).get("ap50_95")
                row.append(_format_number(value))
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("Per-class AP was unavailable in the discovered summaries.")
    lines.extend(["", "## Comparability warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("No differences were detected in the recorded comparability fields among eligible runs.")
    if failed_models:
        lines.extend(["", "## Failed or incomplete runs", ""])
        lines.extend(f"- {name}" for name in failed_models)
    incomplete = [record for record in records if record["status"] != "completed"]
    if incomplete:
        lines.extend(["", "## Incomplete experiments", ""])
        for record in incomplete:
            detail = "; ".join(record.get("warnings", [])) or record["status"]
            lines.append(f"- `{record['run_name']}` ({record['status']}): {detail}")
    smoke = [record["run_name"] for record in records if record["smoke_test"]]
    if smoke:
        lines.extend(["", "## Smoke tests", "", f"Excluded from main ranking by default: {', '.join(smoke)}."])
        if include_smoke:
            lines.append("Smoke tests were included because `--include-smoke-tests` was specified; interpret their scores as diagnostics, not comparable full runs.")
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- F1 is derived as 2 × precision × recall / (precision + recall) using the recorded validation operating point. It is not AP.",
        "- The best epoch is read from `results.csv` using the experiment's recorded checkpoint-selection metric (mAP50-95 for this detection task). Summary/CSV mismatches are reported rather than silently hidden.",
        "- Training curves and loss plots retain each experiment's original epoch log. A single completed experiment is valid input and still produces the plots.",
        "- Runtime and hardware differences can make wall time and latency comparisons non-equivalent.",
        "",
        "## Figures",
        "",
        "- `global_metrics.png`",
        "- `learning_curves.png`",
        "- `loss_curves.png`",
        "- `per_class_ap.png` and `class_model_heatmap.png`",
        "- `training_time.png`",
        "- `map_latency.png`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(runs_dir: Path, output_dir: Path, include_smoke_tests: bool = False) -> Path:
    """Discover, normalize, compare, and report training experiments."""
    runs_dir = runs_dir.expanduser().resolve()
    experiments = discover_experiments(runs_dir)
    records = [experiment.record for experiment in experiments]
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_warnings = _comparison_warnings(records)
    for experiment in experiments:
        comparison_warnings.extend(f"{experiment.run_dir.name}: {warning}." for warning in experiment.warnings)
    failed_models: list[str] = []
    model_comparison_path = runs_dir / "model_comparison.json"
    model_comparison = _read_json(model_comparison_path)
    if model_comparison:
        for entry in model_comparison.get("models", []):
            if isinstance(entry, dict) and entry.get("status") == "failed":
                error_type = entry.get("error_type") or "training failed"
                failed_models.append(f"{entry.get('model', 'unknown')}: {error_type}")

    eligible = [record for record in records if record["eligible_for_ranking"] and (include_smoke_tests or not record["smoke_test"])]
    eligible.sort(key=lambda record: (record.get("map50_95") is not None, record.get("map50_95") or float("-inf")), reverse=True)
    plots = [
        experiment for experiment in experiments
        if experiment.rows and experiment.record["eligible_for_ranking"]
        and (include_smoke_tests or not experiment.smoke_test)
    ]
    _save_comparison_csv(output_dir / "comparison.csv", records)
    payload = {
        "runs_directory": runs_dir.name,
        "ranking_metric": "validation mAP50-95",
        "test_split_used": False,
        "include_smoke_tests": include_smoke_tests,
        "ranked_experiments": [record["run_name"] for record in eligible],
        "experiments": records,
        "failed_models_from_training_summary": failed_models,
        "warnings": comparison_warnings,
    }
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_global(eligible, output_dir)
    _plot_learning_curves(plots, output_dir)
    _plot_class_ap(eligible, output_dir)
    _plot_training_time(eligible, output_dir)
    _plot_accuracy_latency(eligible, output_dir)
    _write_report(output_dir / "report.md", records, comparison_warnings, failed_models, include_smoke_tests)
    print(f"Comparison artifacts saved to: {output_dir}")
    if not experiments:
        print(f"No experiment_summary.json, results.csv, or args.yaml files found under {runs_dir}.")
    return output_dir


def parse_args() -> argparse.Namespace:
    """Define the comparator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS, help="Root containing Ultralytics experiment folders")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Comparison artifact directory")
    parser.add_argument("--include-smoke-tests", action="store_true", help="Include named or <=3-epoch smoke tests in ranking and plots")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    compare(args.runs_dir, args.output_dir, args.include_smoke_tests)


if __name__ == "__main__":
    main()
