#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Validate a YOLO export and create a deterministic annotation preview."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "artifacts" / "yolo_dataset"


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if directory.is_dir() else []


def read_corrections(path: Path, errors: list[str]) -> tuple[dict[tuple[str, str], set[int]], dict[str, int]]:
    corrected_objects: dict[tuple[str, str], set[int]] = {}
    split_counts: Counter[str] = Counter()
    if not path.exists():
        errors.append("bbox_corrections.csv is missing")
        return corrected_objects, dict(split_counts)

    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row_number, row in enumerate(csv.DictReader(stream), start=2):
                split = row.get("split", "")
                image_stem = Path(row.get("image", "")).stem
                try:
                    object_index = int(row.get("object_index", ""))
                except (TypeError, ValueError):
                    errors.append(f"bbox_corrections.csv:{row_number}: invalid object_index")
                    continue
                if split not in SPLITS or not image_stem or object_index < 1:
                    errors.append(f"bbox_corrections.csv:{row_number}: invalid split, image, or object_index")
                    continue
                corrected_objects.setdefault((split, image_stem), set()).add(object_index)
                split_counts[split] += 1
    except (OSError, csv.Error):
        errors.append("Could not read bbox_corrections.csv")
    return corrected_objects, dict(split_counts)


def load_metadata(dataset_dir: Path, errors: list[str]) -> tuple[list[str], dict[str, Any]]:
    report_path = dataset_dir / "conversion_report.json"
    try:
        conversion = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("Could not read conversion_report.json")
        return [], {}

    class_names = conversion.get("class_names")
    if not isinstance(class_names, list) or not class_names or not all(isinstance(name, str) for name in class_names):
        errors.append("conversion_report.json has no valid class_names list")
        class_names = []

    expected_ids = {name: index for index, name in enumerate(class_names)}
    if conversion.get("class_ids") != expected_ids:
        errors.append("conversion_report.json class_ids do not match class_names order")

    classes_path = dataset_dir / "classes.txt"
    try:
        class_file_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if class_file_names != class_names:
            errors.append("classes.txt does not match conversion_report.json class_names")
    except OSError:
        errors.append("Could not read classes.txt")

    yaml_path = dataset_dir / "data.yaml"
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        errors.append("Could not read or parse data.yaml")
        return class_names, conversion

    if not isinstance(data, dict):
        errors.append("data.yaml must contain a mapping")
        return class_names, conversion

    # The export uses relative split paths resolved from the YAML directory.
    if data.get("path"):
        errors.append("data.yaml must resolve split paths from its own directory (omit 'path')")
    for split in SPLITS:
        if data.get(split) != f"images/{split}":
            errors.append(f"data.yaml must set {split} to images/{split}")
    expected_names = dict(enumerate(class_names))
    if data.get("names") not in (class_names, expected_names):
        errors.append("data.yaml names do not match conversion_report.json class order")
    if "nc" in data and data["nc"] != len(class_names):
        errors.append("data.yaml nc does not match the number of classes")

    return class_names, conversion


def scan_split(
    dataset_dir: Path,
    split: str,
    class_names: list[str],
    conversion: dict[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    if not images_dir.is_dir():
        errors.append(f"{split}: images directory is missing")
    if not labels_dir.is_dir():
        errors.append(f"{split}: labels directory is missing")
    images = image_files(images_dir)
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []

    image_stems = [path.stem for path in images]
    label_stems = [path.stem for path in labels]
    if len(image_stems) != len(set(image_stems)):
        errors.append(f"{split}: multiple image files share the same basename")
    image_by_stem = {path.stem: path for path in images}
    label_by_stem = {path.stem: path for path in labels}
    missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
    missing_images = sorted(set(label_by_stem) - set(image_by_stem))
    if missing_labels:
        errors.append(f"{split}: {len(missing_labels)} images have no label file (e.g. {missing_labels[:5]})")
    if missing_images:
        errors.append(f"{split}: {len(missing_images)} label files have no image (e.g. {missing_images[:5]})")

    class_counts: Counter[str] = Counter()
    object_count = 0
    records: dict[str, dict[str, Any]] = {}
    for stem, image_path in image_by_stem.items():
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, UnidentifiedImageError):
            errors.append(f"{split}/{image_path.name}: unreadable image")
            continue

        label_path = label_by_stem.get(stem)
        detections: list[dict[str, Any]] = []
        if label_path:
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                errors.append(f"{split}/{label_path.name}: could not read label file")
                lines = []
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                fields = line.split()
                location = f"{split}/{label_path.name}:{line_number}"
                if len(fields) != 5:
                    errors.append(f"{location}: expected 5 fields, found {len(fields)}")
                    continue
                try:
                    class_id = int(fields[0])
                    cx, cy, box_width, box_height = map(float, fields[1:])
                except ValueError:
                    errors.append(f"{location}: non-numeric class or box values")
                    continue
                values = (cx, cy, box_width, box_height)
                if class_id < 0 or class_id >= len(class_names):
                    errors.append(f"{location}: class id {class_id} is outside the class list")
                    continue
                if not all(math.isfinite(value) for value in values):
                    errors.append(f"{location}: box contains a non-finite value")
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    errors.append(f"{location}: box center is outside [0, 1]")
                    continue
                if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
                    errors.append(f"{location}: box width or height is outside (0, 1]")
                    continue
                if cx - box_width / 2.0 < -1e-6 or cx + box_width / 2.0 > 1.0 + 1e-6:
                    errors.append(f"{location}: box extends beyond the normalized image width")
                    continue
                if cy - box_height / 2.0 < -1e-6 or cy + box_height / 2.0 > 1.0 + 1e-6:
                    errors.append(f"{location}: box extends beyond the normalized image height")
                    continue

                class_counts[class_names[class_id]] += 1
                object_count += 1
                detections.append({
                    "class_id": class_id,
                    "cx": cx,
                    "cy": cy,
                    "width": box_width,
                    "height": box_height,
                    "object_index": line_number,
                    "image_width": image_width,
                    "image_height": image_height,
                })
        records[stem] = {"path": image_path, "detections": detections, "split": split}

    expected = conversion.get("splits", {}).get(split, {})
    if len(images) != expected.get("images"):
        errors.append(f"{split}: found {len(images)} images, expected {expected.get('images')}")
    if object_count != expected.get("objects"):
        errors.append(f"{split}: found {object_count} objects, expected {expected.get('objects')}")
    expected_classes = expected.get("class_instances", {})
    actual_classes = {name: class_counts.get(name, 0) for name in class_names}
    if actual_classes != expected_classes:
        errors.append(f"{split}: class counts do not match conversion_report.json")

    result = {
        "images": len(images),
        "label_files": len(labels),
        "objects": object_count,
        "class_instances": actual_classes,
        "clipped_boxes": 0,
    }
    return result, records


def read_correction_rows(path: Path, errors: list[str]) -> tuple[dict[tuple[str, str], set[int]], dict[str, int]]:
    corrected_objects, split_counts = read_corrections(path, errors)
    return corrected_objects, split_counts


def create_preview(
    dataset_dir: Path,
    class_names: list[str],
    split_records: dict[str, dict[str, dict[str, Any]]],
    corrected_objects: dict[tuple[str, str], set[int]],
    preview_limit: int,
) -> tuple[str, int]:
    selected: list[tuple[str, str]] = []

    def add(key: tuple[str, str]) -> None:
        if key not in selected and key[1] in split_records.get(key[0], {}):
            selected.append(key)

    # Prioritize every image with a clipped box so the edge cases are reviewable.
    for split, stem in sorted(corrected_objects):
        add((split, stem))

    # Add one smallest-object example for each class.
    for class_id in range(len(class_names)):
        candidates: list[tuple[float, str, str]] = []
        for split, records in split_records.items():
            for stem, record in records.items():
                for detection in record["detections"]:
                    if detection["class_id"] == class_id:
                        area = detection["width"] * detection["height"]
                        candidates.append((area, split, stem))
        if candidates:
            _, split, stem = min(candidates)
            add((split, stem))

    # Add the most crowded image from each split.
    for split, records in split_records.items():
        if records:
            stem = max(records, key=lambda name: len(records[name]["detections"]))
            add((split, stem))

    selected = selected[:preview_limit]
    if not selected:
        return "annotation_preview.jpg", 0

    columns = 4
    cell_width, cell_height = 500, 340
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
        (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    ]
    corrected_keys = {
        (split, stem, object_index)
        for (split, stem), object_indexes in corrected_objects.items()
        for object_index in object_indexes
    }

    for index, (split, stem) in enumerate(selected):
        record = split_records[split][stem]
        image_path: Path = record["path"]
        with Image.open(image_path) as image:
            thumb = image.convert("RGB")
        original_size = thumb.size
        thumb.thumbnail((cell_width - 16, cell_height - 46))
        x_offset = (index % columns) * cell_width + 8
        y_offset = (index // columns) * cell_height + 32
        canvas.paste(thumb, (x_offset, y_offset))
        draw.text((x_offset, y_offset - 22), f"{split}/{image_path.name} · {len(record['detections'])} objetos", fill="black", font=font)

        scale_x = thumb.width / original_size[0]
        scale_y = thumb.height / original_size[1]
        for detection in record["detections"]:
            class_id = detection["class_id"]
            box_width = detection["width"] * original_size[0]
            box_height = detection["height"] * original_size[1]
            x1 = (detection["cx"] * original_size[0] - box_width / 2) * scale_x + x_offset
            y1 = (detection["cy"] * original_size[1] - box_height / 2) * scale_y + y_offset
            x2 = (detection["cx"] * original_size[0] + box_width / 2) * scale_x + x_offset
            y2 = (detection["cy"] * original_size[1] + box_height / 2) * scale_y + y_offset
            is_corrected = (split, stem, detection["object_index"]) in corrected_keys
            color = (220, 0, 0) if is_corrected else palette[class_id % len(palette)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3 if is_corrected else 2)
            label = class_names[class_id] + (" *" if is_corrected else "")
            label_width = draw.textbbox((0, 0), label, font=font)[2]
            label_y = max(y_offset, y1 - 14)
            draw.rectangle((x1, label_y, x1 + label_width + 4, label_y + 13), fill=color)
            draw.text((x1 + 2, label_y + 1), label, fill="white", font=font)

    preview_path = dataset_dir / "annotation_preview.jpg"
    canvas.save(preview_path, quality=90, optimize=True)
    return preview_path.name, len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a converted YOLO dataset and create annotation previews.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET, help="YOLO dataset directory to inspect")
    parser.add_argument("--preview-limit", type=int, default=60, help="Maximum number of images in the contact sheet")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if args.preview_limit < 1:
        raise SystemExit("--preview-limit must be at least 1")
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir.name}")

    errors: list[str] = []
    class_names, conversion = load_metadata(dataset_dir, errors)
    corrected_objects, clipped_per_split = read_correction_rows(dataset_dir / "bbox_corrections.csv", errors)
    split_results: dict[str, dict[str, Any]] = {}
    split_records: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        result, records = scan_split(dataset_dir, split, class_names, conversion, errors)
        result["clipped_boxes"] = clipped_per_split.get(split, 0)
        expected_clipped = conversion.get("splits", {}).get(split, {}).get("clipped_boxes")
        if expected_clipped is not None and result["clipped_boxes"] != expected_clipped:
            errors.append(f"{split}: correction rows do not match conversion_report.json")
        split_results[split] = result
        split_records[split] = records

    total_clipped = sum(clipped_per_split.values())
    expected_total_clipped = conversion.get("total_clipped_boxes")
    if expected_total_clipped is not None and total_clipped != expected_total_clipped:
        errors.append("bbox_corrections.csv total does not match conversion_report.json")

    preview_name, preview_count = create_preview(
        dataset_dir, class_names, split_records, corrected_objects, args.preview_limit
    )
    validation = {
        "dataset": dataset_dir.name,
        "passed": not errors,
        "classes": class_names,
        "splits": split_results,
        "total_images": sum(item["images"] for item in split_results.values()),
        "total_objects": sum(item["objects"] for item in split_results.values()),
        "total_clipped_boxes": total_clipped,
        "preview": preview_name,
        "preview_images": preview_count,
        "errors": errors,
    }
    report_path = dataset_dir / "validation_report.json"
    report_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Validation {'passed' if not errors else 'failed'} for {dataset_dir.name}")
    print(f"Images: {validation['total_images']}; objects: {validation['total_objects']}; clipped boxes: {total_clipped}")
    print(f"Report: {report_path.name}; preview: {preview_name} ({preview_count} images)")
    if errors:
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
