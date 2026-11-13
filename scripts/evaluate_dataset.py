#!/usr/bin/env python3
"""Audit the CODD image-detection dataset and write a compact EDA report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, UnidentifiedImageError


SPLITS = ("training", "validation", "testing")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "Construction and Demolition Waste Object Detection Dataset  (CODD)"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "dataset_audit"


def parse_bool(value: str | None) -> bool:
    return str(value or "0").strip().lower() in {"1", "true", "yes"}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(split: str, directory: Path) -> dict[str, Any]:
    images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    annotations = sorted(directory.glob("*.xml"))
    image_by_stem = {p.stem: p for p in images}
    annotation_by_stem = {p.stem: p for p in annotations}
    missing_xml = sorted(set(image_by_stem) - set(annotation_by_stem))
    orphan_xml = sorted(set(annotation_by_stem) - set(image_by_stem))

    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    boxes_by_image: dict[str, list[dict[str, Any]]] = {}
    issues: list[dict[str, str]] = []
    class_instances: Counter[str] = Counter()
    class_images: Counter[str] = Counter()
    image_hashes: dict[str, str] = {}
    object_counts: list[int] = []
    dimensions: Counter[str] = Counter()
    total_image_bytes = 0

    for image_path in images:
        total_image_bytes += image_path.stat().st_size
        width = height = 0
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
            dimensions[f"{width}x{height}"] += 1
            image_hashes[image_path.name] = image_sha256(image_path)
        except (OSError, UnidentifiedImageError) as exc:
            issues.append({"split": split, "file": image_path.name, "issue": f"Unreadable image: {exc}"})

        annotation_path = annotation_by_stem.get(image_path.stem)
        per_image: Counter[str] = Counter()
        image_boxes: list[dict[str, Any]] = []
        annotation_width = annotation_height = 0

        if annotation_path is None:
            issues.append({"split": split, "file": image_path.name, "issue": "Missing XML annotation"})
        else:
            try:
                root = ET.parse(annotation_path).getroot()
                declared_name = Path(root.findtext("filename", "")).name
                if declared_name and declared_name != image_path.name:
                    issues.append({"split": split, "file": annotation_path.name, "issue": f"XML filename says {declared_name}"})
                size = root.find("size")
                if size is not None:
                    annotation_width = int(size.findtext("width", "0") or 0)
                    annotation_height = int(size.findtext("height", "0") or 0)
                if width and height and (annotation_width, annotation_height) != (width, height):
                    issues.append({"split": split, "file": annotation_path.name, "issue": f"XML dimensions {annotation_width}x{annotation_height} != image {width}x{height}"})

                for index, obj in enumerate(root.findall("object"), start=1):
                    label = obj.findtext("name", "unknown").strip() or "unknown"
                    per_image[label] += 1
                    class_instances[label] += 1
                    difficult = parse_bool(obj.findtext("difficult"))
                    truncated = parse_bool(obj.findtext("truncated"))
                    occluded = parse_bool(obj.findtext("occluded"))
                    bounds = obj.find("bndbox")
                    if bounds is None:
                        issues.append({"split": split, "file": annotation_path.name, "issue": f"Object {index} ({label}) has no bndbox"})
                        continue
                    try:
                        xmin = float(bounds.findtext("xmin", "nan"))
                        ymin = float(bounds.findtext("ymin", "nan"))
                        xmax = float(bounds.findtext("xmax", "nan"))
                        ymax = float(bounds.findtext("ymax", "nan"))
                    except (TypeError, ValueError):
                        issues.append({"split": split, "file": annotation_path.name, "issue": f"Object {index} ({label}) has non-numeric coordinates"})
                        continue

                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    valid_area = box_width > 0 and box_height > 0
                    outside = (
                        not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax))
                        or xmin < 0 or ymin < 0
                        or (width and xmax > width) or (height and ymax > height)
                    )
                    if not valid_area:
                        issues.append({"split": split, "file": annotation_path.name, "issue": f"Object {index} ({label}) has degenerate bbox"})
                    if outside:
                        issues.append({"split": split, "file": annotation_path.name, "issue": f"Object {index} ({label}) bbox is outside image bounds"})

                    polygon = obj.find("polygon")
                    polygon_points = len([child for child in polygon or [] if child.tag.startswith("x")]) if polygon is not None else 0
                    image_area = width * height if width and height else 0
                    area_pct = (box_width * box_height / image_area * 100) if valid_area and image_area else 0.0
                    row = {
                        "split": split,
                        "image": image_path.name,
                        "object_index": index,
                        "class": label,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "width_fraction_pct": box_width / width * 100 if valid_area and width else 0.0,
                        "height_fraction_pct": box_height / height * 100 if valid_area and height else 0.0,
                        "area_fraction_pct": area_pct,
                        "bbox_out_of_bounds": outside,
                        "polygon_points": polygon_points,
                        "difficult": difficult,
                        "truncated": truncated,
                        "occluded": occluded,
                    }
                    box_rows.append(row)
                    image_boxes.append(row)
            except (ET.ParseError, OSError, ValueError) as exc:
                issues.append({"split": split, "file": annotation_path.name, "issue": f"Invalid XML: {exc}"})

        class_images.update(per_image.keys())
        object_counts.append(sum(per_image.values()))
        boxes_by_image[image_path.name] = image_boxes
        image_rows.append({
            "split": split,
            "image": image_path.name,
            "width": width,
            "height": height,
            "objects": sum(per_image.values()),
            "classes_present": ";".join(sorted(per_image)),
            "xml": annotation_path.name if annotation_path else "",
        })

    for orphan in orphan_xml:
        issues.append({"split": split, "file": f"{orphan}.xml", "issue": "XML has no matching image"})

    instance_total = sum(class_instances.values())
    return {
        "split": split,
        "image_paths": images,
        "image_boxes": boxes_by_image,
        "image_rows": image_rows,
        "box_rows": box_rows,
        "issues": issues,
        "image_hashes": image_hashes,
        "class_instances": class_instances,
        "class_images": class_images,
        "object_counts": object_counts,
        "dimensions": dimensions,
        "metrics": {
            "images": len(images),
            "xml_annotations": len(annotations),
            "objects": instance_total,
            "paired_images": len(images) - len(missing_xml),
            "missing_xml": len(missing_xml),
            "orphan_xml": len(orphan_xml),
            "mean_objects_per_image": round(mean(object_counts), 3) if object_counts else 0,
            "median_objects_per_image": median(object_counts) if object_counts else 0,
            "min_objects_per_image": min(object_counts, default=0),
            "max_objects_per_image": max(object_counts, default=0),
            "image_dimensions": dict(dimensions),
            "image_bytes": total_image_bytes,
            "classes": len(class_instances),
        },
    }


def create_plots(results: dict[str, dict[str, Any]], classes: list[str], out_dir: Path) -> None:
    colors = {"training": "#2878B5", "validation": "#F28E2B", "testing": "#59A14F"}
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = list(range(len(classes)))
    bar_height = 0.24
    for idx, split in enumerate(SPLITS):
        counts = results[split]["class_instances"]
        offset = (idx - 1) * bar_height
        total = max(results[split]["metrics"]["objects"], 1)
        shares = [counts.get(c, 0) / total * 100 for c in classes]
        ax.barh([p + offset for p in positions], shares, height=bar_height,
                color=colors[split], label=split.capitalize())
    ax.set_yticks(positions, classes)
    ax.invert_yaxis()
    ax.set_xlabel("Share of annotated objects (%)")
    ax.set_title("Relative class distribution by split")
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    max_objects = max((max(r["object_counts"], default=0) for r in results.values()), default=1)
    bins = range(0, max_objects + 2)
    for split in SPLITS:
        ax.hist(results[split]["object_counts"], bins=bins, alpha=0.55, color=colors[split], label=split.capitalize())
    ax.set_xlabel("Objects in one image")
    ax.set_ylabel("Number of images")
    ax.set_title("Objects per image by split")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "objects_per_image.png", bbox_inches="tight")
    plt.close(fig)

    boxes = [box for result in results.values() for box in result["box_rows"] if not box["bbox_out_of_bounds"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if boxes:
        x = [max(box["width_fraction_pct"], 1e-4) for box in boxes]
        y = [max(box["height_fraction_pct"], 1e-4) for box in boxes]
        axes[0].scatter(x, y, s=7, alpha=0.16, color="#2878B5", edgecolors="none")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Box width (% of image)")
        axes[0].set_ylabel("Box height (% of image)")
        areas = [max(box["area_fraction_pct"], 1e-5) for box in boxes]
        log_min = math.floor(math.log10(min(areas)))
        log_max = math.ceil(math.log10(max(areas)))
        area_bins = [10 ** (log_min + i * (log_max - log_min) / 40) for i in range(41)]
        axes[1].hist(areas, bins=area_bins, color="#7B61A8", alpha=0.85)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Bounding-box area (% of image, log scale)")
        axes[1].set_ylabel("Objects")
    axes[0].set_title("Bounding-box geometry")
    axes[1].set_title("Object area distribution")
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "bbox_geometry.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(SPLITS), figsize=(15, 4.5))
    palette = plt.get_cmap("tab20")
    class_colors = {c: palette(i % 20) for i, c in enumerate(classes)}
    for ax, split in zip(axes, SPLITS):
        candidates = []
        for image_path in results[split]["image_paths"]:
            boxes_for_image = results[split]["image_boxes"].get(image_path.name, [])
            if 2 <= len(boxes_for_image) <= 6:
                candidates.append((image_path, boxes_for_image))
        if not candidates:
            ax.axis("off")
            continue
        image_path, boxes_for_image = max(candidates, key=lambda item: (len({b["class"] for b in item[1]}), -abs(len(item[1]) - 4)))
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))
        for box in boxes_for_image:
            color = class_colors[box["class"]]
            ax.add_patch(Rectangle((box["xmin"], box["ymin"]), box["xmax"] - box["xmin"], box["ymax"] - box["ymin"],
                                   fill=False, edgecolor=color, linewidth=1.4))
            ax.text(box["xmin"], max(0, box["ymin"] - 3), box["class"], color="white", fontsize=6,
                    bbox={"facecolor": color, "alpha": 0.85, "pad": 1, "edgecolor": "none"})
        ax.set_title(f"{split.capitalize()} · {image_path.name}")
        ax.axis("off")
    fig.suptitle("Example annotations (2–6 objects per image)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "annotated_examples.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a quality and distribution report for the CODD dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET, help="CODD directory containing training/validation/testing")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for CSV, JSON, Markdown and PNG outputs")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        dataset_reference = dataset_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Keep generated artifacts portable when a custom dataset lives elsewhere.
        dataset_reference = dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        directory = dataset_dir / split
        if not directory.is_dir():
            raise SystemExit(f"Missing split directory: {directory}")
        results[split] = inspect_split(split, directory)

    classes = sorted({label for result in results.values() for label in result["class_instances"]})
    all_issues = [issue for result in results.values() for issue in result["issues"]]
    total_images = sum(result["metrics"]["images"] for result in results.values())
    total_objects = sum(result["metrics"]["objects"] for result in results.values())

    class_rows = []
    for label in classes:
        for split in SPLITS:
            class_rows.append({
                "split": split,
                "class": label,
                "instances": results[split]["class_instances"].get(label, 0),
                "images_containing_class": results[split]["class_images"].get(label, 0),
                "instance_share_pct": round(results[split]["class_instances"].get(label, 0) / max(results[split]["metrics"]["objects"], 1) * 100, 3),
            })
    image_rows = [row for split in SPLITS for row in results[split]["image_rows"]]
    box_rows = [row for split in SPLITS for row in results[split]["box_rows"]]
    issue_rows = all_issues

    duplicate_hashes: dict[str, list[str]] = defaultdict(list)
    stem_splits: dict[str, set[str]] = defaultdict(set)
    for split, result in results.items():
        for filename, digest in result["image_hashes"].items():
            duplicate_hashes[digest].append(f"{split}/{filename}")
            stem_splits[Path(filename).stem].add(split)
    exact_duplicate_groups = [paths for paths in duplicate_hashes.values() if len(paths) > 1]
    repeated_stems = {stem: sorted(splits) for stem, splits in stem_splits.items() if len(splits) > 1}

    bbox_outside = sum(bool(box["bbox_out_of_bounds"]) for box in box_rows)
    summary = {
        "dataset": dataset_reference,
        "splits": {split: result["metrics"] for split, result in results.items()},
        "total_images": total_images,
        "total_objects": total_objects,
        "classes": classes,
        "total_classes": len(classes),
        "exact_duplicate_groups_across_splits": exact_duplicate_groups,
        "basenames_reused_across_splits": repeated_stems,
        "issue_counts": dict(Counter(issue["issue"].split(":")[0] for issue in all_issues)),
        "total_issues": len(all_issues),
        "bbox_out_of_bounds": bbox_outside,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "class_distribution.csv", class_rows, ["split", "class", "instances", "images_containing_class", "instance_share_pct"])
    write_csv(output_dir / "image_inventory.csv", image_rows, ["split", "image", "width", "height", "objects", "classes_present", "xml"])
    write_csv(output_dir / "bbox_inventory.csv", box_rows, ["split", "image", "object_index", "class", "xmin", "ymin", "xmax", "ymax", "width_fraction_pct", "height_fraction_pct", "area_fraction_pct", "bbox_out_of_bounds", "polygon_points", "difficult", "truncated", "occluded"])
    write_csv(output_dir / "data_issues.csv", issue_rows, ["split", "file", "issue"])
    create_plots(results, classes, output_dir)

    lines = [
        "# CODD dataset audit",
        "",
        f"- Dataset: `{dataset_reference}`",
        f"- Images: **{total_images:,}**",
        f"- Annotated objects: **{total_objects:,}**",
        f"- Classes: **{len(classes)}** ({', '.join(classes)})",
        f"- Exact duplicate groups across splits: **{len(exact_duplicate_groups)}**",
        f"- Basenames reused across splits: **{len(repeated_stems)}** (keep split folders separate)",
        f"- Data-quality issues recorded: **{len(all_issues)}**",
        "",
        "## Split summary",
        "",
        "| Split | Images | XML | Objects | Mean objects/image | Dimensions |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in SPLITS:
        metrics = results[split]["metrics"]
        dims = ", ".join(f"{size}: {count}" for size, count in sorted(metrics["image_dimensions"].items())) or "n/a"
        lines.append(f"| {split} | {metrics['images']:,} | {metrics['xml_annotations']:,} | {metrics['objects']:,} | {metrics['mean_objects_per_image']:.2f} | {dims} |")
    lines += [
        "",
        "## Class counts",
        "",
        "| Class | Training | Validation | Testing |",
        "|---|---:|---:|---:|",
    ]
    for label in classes:
        values = [results[split]["class_instances"].get(label, 0) for split in SPLITS]
        lines.append(f"| {label} | {values[0]:,} | {values[1]:,} | {values[2]:,} |")
    lines += [
        "",
        "## Figures",
        "",
        "- `class_distribution.png`: object instances and class balance by split.",
        "- `objects_per_image.png`: image density by split.",
        "- `bbox_geometry.png`: relative box sizes and area distribution.",
        "- `annotated_examples.png`: visual spot-check of example boxes.",
        "",
        "## Data quality",
        "",
        f"See `data_issues.csv` for per-file issues. **{bbox_outside} / {total_objects:,} boxes ({bbox_outside / max(total_objects, 1) * 100:.2f}%) extend beyond image bounds.** Review these border boxes when converting from PASCAL VOC coordinates to the training framework’s convention.",
        "",
        "## Evaluation guidance",
        "",
        "For object detection, report mAP@0.5 and mAP@0.5:0.95 plus per-class precision/recall. Avoid accuracy as the headline metric. Keep `testing` untouched until the final evaluation; use `validation` for model selection.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit written to: {output_dir}")
    print(f"Images: {total_images}; objects: {total_objects}; classes: {len(classes)}; issues: {len(all_issues)}")


if __name__ == "__main__":
    main()
