#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run YOLO inference on an image or a flat directory of images."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "detect" / "codd-yolov8n-img640-seed42" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "predictions"
PINNED_ULTRALYTICS_VERSION = "8.4.210"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory containing images")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Trained YOLO .pt checkpoint")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for annotated images and predictions.json")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: final evaluation size 640)")
    parser.add_argument("--conf", type=float, default=0.50, help="Minimum confidence (default: validation-selected 0.50)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold (default: validation-selected 0.50)")
    parser.add_argument("--device", default=None, help="Ultralytics device, for example 0 or cpu (default: auto)")
    return parser.parse_args()


def collect_images(source: Path) -> tuple[list[Path], Path]:
    """Return supported images and the base used for portable image names."""
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"Unsupported image extension: {source.suffix}")
        return [source], source.parent
    if not source.is_dir():
        raise SystemExit(f"Source does not exist or is not a file/directory: {source.name}")
    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No supported images found directly inside: {source}")
    return images, source


def json_safe(value: Any) -> Any:
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


def run_prediction(args: argparse.Namespace) -> Path:
    if args.imgsz < 32:
        raise SystemExit("imgsz must be at least 32 pixels.")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise SystemExit("conf and iou must be in [0, 1].")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise SystemExit(f"Checkpoint does not exist or is not a .pt file: {checkpoint.name}")
    images, base_dir = collect_images(args.source)

    try:
        installed = version("ultralytics")
    except PackageNotFoundError as exc:
        raise SystemExit("Ultralytics is not installed; install one of the project requirements profiles first.") from exc
    if installed != PINNED_ULTRALYTICS_VERSION:
        raise SystemExit(f"Expected ultralytics=={PINNED_ULTRALYTICS_VERSION}, found {installed}.")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("PyTorch or Ultralytics could not be imported.") from exc

    model = YOLO(str(checkpoint))
    names = getattr(model, "names", {})
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    names = {int(class_id): str(name) for class_id, name in names.items()}

    output_dir = args.output_dir.expanduser().resolve()
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    image_results: list[dict[str, Any]] = []

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            save=False,
        )[0]

        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confidences = boxes.conf.detach().cpu().numpy()
            for box, class_id, confidence in zip(xyxy, class_ids, confidences):
                detections.append({
                    "class_id": int(class_id),
                    "class_name": names.get(int(class_id), str(class_id)),
                    "confidence": round(float(confidence), 6),
                    "bbox_xyxy_pixels": [round(float(value), 2) for value in box],
                })

        # Ultralytics returns the rendered array in BGR channel order.
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1].copy()
        annotated_path = annotated_dir / image_path.name
        Image.fromarray(annotated_rgb).save(annotated_path)

        try:
            relative_image = image_path.relative_to(base_dir).as_posix()
        except ValueError:
            relative_image = image_path.name
        image_results.append({
            "image": relative_image,
            "width": int(result.orig_shape[1]),
            "height": int(result.orig_shape[0]),
            "annotated_image": f"annotated/{annotated_path.name}",
            "detections": detections,
        })

    experiment_name = checkpoint.parent.parent.name if checkpoint.parent.name == "weights" else checkpoint.stem
    model_match = re.search(r"(yolov8|yolo11|yolo26)([nsmlx])", experiment_name.lower())
    summary = {
        "model": "".join(model_match.groups()) if model_match else experiment_name,
        "experiment": experiment_name if model_match else None,
        "checkpoint": checkpoint.name,
        "ultralytics": installed,
        "torch": str(torch.__version__),
        "settings": {
            "imgsz": args.imgsz,
            "confidence": args.conf,
            "nms_iou": args.iou,
            "device": args.device or "auto",
        },
        "images_processed": len(image_results),
        "detections_total": sum(len(item["detections"]) for item in image_results),
        "images": image_results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_path = output_dir / "predictions.json"
    results_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Images processed: {len(image_results)}")
    print(f"Detections: {summary['detections_total']}")
    for label, path in (("Annotated images", annotated_dir), ("Results JSON", results_path)):
        try:
            display_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display_path = path.name
        print(f"{label}: {display_path}")
    return results_path


def main() -> None:
    run_prediction(parse_args())


if __name__ == "__main__":
    main()
