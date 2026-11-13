#!/usr/bin/env python3
"""Convert CODD Pascal VOC XML annotations to YOLO detection format."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SPLITS = {"training": "train", "validation": "val", "testing": "test"}
CLASS_NAMES = [
    "brick",
    "concrete",
    "foam",
    "general_w",
    "gypsum_board",
    "pipes",
    "plastic",
    "stone",
    "tile",
    "wood",
]
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "Construction and Demolition Waste Object Detection Dataset  (CODD)"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "yolo_dataset"


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def portable_path(path: Path) -> str:
    """Return a repository-relative path, or only the basename for external paths."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def voc_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, image_width: int, image_height: int
) -> tuple[float, float, float, float, tuple[float, float, float, float]]:
    """Convert 1-based inclusive Pascal VOC boxes to normalized YOLO xywh.

    VOC's minimum coordinates are 1-based; subtract one to get zero-based image
    coordinates. The maximum coordinates become pixel-edge coordinates. Clip
    both ends to the image extent before normalization.
    """
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError("bbox coordinates must be finite numbers")

    unclipped = (xmin - 1.0, ymin - 1.0, xmax, ymax)
    x1 = clamp(unclipped[0], 0.0, float(image_width))
    y1 = clamp(unclipped[1], 0.0, float(image_height))
    x2 = clamp(unclipped[2], 0.0, float(image_width))
    y2 = clamp(unclipped[3], 0.0, float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox collapses after clipping: {(xmin, ymin, xmax, ymax)}")

    center_x = ((x1 + x2) / 2.0) / image_width
    center_y = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return center_x, center_y, width, height, (x1, y1, x2, y2)


def scan_split(source_dir: Path, split: str) -> dict[str, Any]:
    images_dir = source_dir / split
    if not images_dir.is_dir():
        raise ValueError(f"Missing source split directory: {split}")

    images = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    annotations = sorted(images_dir.glob("*.xml"))
    image_by_stem = {path.stem: path for path in images}
    annotation_by_stem = {path.stem: path for path in annotations}

    missing_xml = sorted(set(image_by_stem) - set(annotation_by_stem))
    orphan_xml = sorted(set(annotation_by_stem) - set(image_by_stem))
    if missing_xml or orphan_xml:
        raise ValueError(
            f"{split}: image/XML pairs do not match "
            f"(images without XML: {missing_xml[:5]}, XML without image: {orphan_xml[:5]})"
        )

    plans: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    class_instances: Counter[str] = Counter()

    for image_path in images:
        annotation_path = annotation_by_stem[image_path.stem]
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Cannot read image {split}/{image_path.name}: {exc}") from exc

        try:
            root = ET.parse(annotation_path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"Cannot parse annotation {split}/{annotation_path.name}: {exc}") from exc

        declared_filename = Path(root.findtext("filename", "")).name
        if declared_filename and declared_filename != image_path.name:
            raise ValueError(f"{split}/{annotation_path.name}: filename is {declared_filename}, expected {image_path.name}")

        size = root.find("size")
        if size is None:
            raise ValueError(f"{split}/{annotation_path.name}: missing <size>")
        declared_width = int(size.findtext("width", "0") or 0)
        declared_height = int(size.findtext("height", "0") or 0)
        if (declared_width, declared_height) != (image_width, image_height):
            raise ValueError(
                f"{split}/{annotation_path.name}: XML dimensions {declared_width}x{declared_height} "
                f"do not match image dimensions {image_width}x{image_height}"
            )

        label_lines: list[str] = []
        for object_index, obj in enumerate(root.findall("object"), start=1):
            class_name = obj.findtext("name", "").strip()
            if class_name not in CLASS_IDS:
                raise ValueError(f"{split}/{annotation_path.name}: unknown class {class_name!r}")
            bounds = obj.find("bndbox")
            if bounds is None:
                raise ValueError(f"{split}/{annotation_path.name}: object {object_index} has no <bndbox>")
            try:
                xmin, ymin, xmax, ymax = (
                    float(bounds.findtext(name, "nan")) for name in ("xmin", "ymin", "xmax", "ymax")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{split}/{annotation_path.name}: object {object_index} has invalid bbox coordinates") from exc

            try:
                cx, cy, width, height, clipped = voc_to_yolo(
                    xmin, ymin, xmax, ymax, image_width, image_height
                )
            except ValueError as exc:
                raise ValueError(f"{split}/{annotation_path.name}: object {object_index} ({class_name}): {exc}") from exc

            class_instances[class_name] += 1
            label_lines.append(f"{CLASS_IDS[class_name]} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")

            original = (xmin, ymin, xmax, ymax)
            converted_pixels = (xmin - 1.0, ymin - 1.0, xmax, ymax)
            if any(abs(before - after) > 1e-9 for before, after in zip(converted_pixels, clipped)):
                corrections.append({
                    "split": SPLITS[split],
                    "image": image_path.name,
                    "class": class_name,
                    "object_index": object_index,
                    "voc_xmin": xmin,
                    "voc_ymin": ymin,
                    "voc_xmax": xmax,
                    "voc_ymax": ymax,
                    "clipped_xmin": clipped[0],
                    "clipped_ymin": clipped[1],
                    "clipped_xmax": clipped[2],
                    "clipped_ymax": clipped[3],
                    "image_width": image_width,
                    "image_height": image_height,
                })

        plans.append({"image": image_path, "label_lines": label_lines})

    return {
        "plans": plans,
        "corrections": corrections,
        "class_instances": dict(class_instances),
        "images": len(images),
        "objects": sum(class_instances.values()),
    }


def write_data_yaml(path: Path) -> None:
    lines = [
        "# With no explicit root, Ultralytics resolves paths from this YAML's directory.",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {class_id}: {name}" for name, class_id in CLASS_IDS.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_dataset(source_dir: Path, output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    backup_dir: Path | None = None
    try:
        for source_split, yolo_split in SPLITS.items():
            images_out = temp_dir / "images" / yolo_split
            labels_out = temp_dir / "labels" / yolo_split
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)
            for plan in results[source_split]["plans"]:
                image_path: Path = plan["image"]
                shutil.copy2(image_path, images_out / image_path.name)
                label_path = labels_out / f"{image_path.stem}.txt"
                label_path.write_text("\n".join(plan["label_lines"]) + ("\n" if plan["label_lines"] else ""), encoding="utf-8")

        write_data_yaml(temp_dir / "data.yaml")
        (temp_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

        correction_rows = [row for split in SPLITS for row in results[split]["corrections"]]
        correction_fields = [
            "split", "image", "class", "object_index", "voc_xmin", "voc_ymin", "voc_xmax", "voc_ymax",
            "clipped_xmin", "clipped_ymin", "clipped_xmax", "clipped_ymax", "image_width", "image_height",
        ]
        with (temp_dir / "bbox_corrections.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=correction_fields)
            writer.writeheader()
            writer.writerows(correction_rows)

        summary = {
            "source": portable_path(source_dir),
            "format": "YOLO object detection (class_id x_center y_center width height; normalized 0-1)",
            "coordinate_conversion": "PASCAL VOC 1-based inclusive to zero-based pixel edges, clipped to image bounds",
            "class_names": CLASS_NAMES,
            "class_ids": CLASS_IDS,
            "splits": {
                SPLITS[source_split]: {
                    "images": results[source_split]["images"],
                    "objects": results[source_split]["objects"],
                    "clipped_boxes": len(results[source_split]["corrections"]),
                    "class_instances": results[source_split]["class_instances"],
                }
                for source_split in SPLITS
            },
            "total_images": sum(result["images"] for result in results.values()),
            "total_objects": sum(result["objects"] for result in results.values()),
            "total_clipped_boxes": len(correction_rows),
        }
        (temp_dir / "conversion_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        # Refuse accidental overwrite. With --overwrite, keep the previous export
        # until the replacement has been fully prepared in the temporary folder.
        if output_dir.exists():
            backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
            if backup_dir.exists():
                raise FileExistsError(f"Backup directory already exists: {backup_dir.name}")
            output_dir.rename(backup_dir)
        try:
            temp_dir.rename(output_dir)
        except OSError:
            # Some Windows setups temporarily block directory renames (for
            # example while indexing files). Fall back to a copy from the
            # completed staging directory, restoring the backup on failure.
            try:
                shutil.copytree(temp_dir, output_dir)
            except Exception:
                if output_dir.exists():
                    shutil.rmtree(output_dir, ignore_errors=True)
                if backup_dir and backup_dir.exists() and not output_dir.exists():
                    backup_dir.rename(output_dir)
                raise
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clipped, normalized YOLO copy of CODD.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE, help="CODD directory with training/validation/testing")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output YOLO dataset directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory after successful conversion")
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Dataset source directory does not exist: {portable_path(source_dir)}")
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise SystemExit("Source and output directories must be separate, non-nested directories.")
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {portable_path(output_dir)}. Choose another --output-dir or pass --overwrite.")

    results = {split: scan_split(source_dir, split) for split in SPLITS}
    found_classes = {name for result in results.values() for name in result["class_instances"]}
    if found_classes != set(CLASS_NAMES):
        missing = sorted(set(CLASS_NAMES) - found_classes)
        unexpected = sorted(found_classes - set(CLASS_NAMES))
        raise SystemExit(f"Class set does not match CODD mapping (missing={missing}, unexpected={unexpected})")

    export_dataset(source_dir, output_dir, results)
    total_images = sum(result["images"] for result in results.values())
    total_objects = sum(result["objects"] for result in results.values())
    total_clipped = sum(len(result["corrections"]) for result in results.values())
    print(f"YOLO dataset written to: {portable_path(output_dir)}")
    print(f"Images: {total_images}; objects: {total_objects}; clipped boxes: {total_clipped}")
    print(f"Class mapping and split counts: {portable_path(output_dir / 'conversion_report.json')}")


if __name__ == "__main__":
    main()
