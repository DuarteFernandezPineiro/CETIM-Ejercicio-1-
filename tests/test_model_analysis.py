# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only unit tests for the YOLO evaluation and experiment analysis helpers."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.compare_experiments import best_epoch_from_results, calculate_f1, compare, discover_experiments, load_experiment
from scripts.evaluate_model import (
    _duplicate_label_rows,
    _make_confusion_matrix,
    box_iou_matrix,
    density_bin,
    f1_score,
    match_image,
    overlap_bins,
    predict_in_batches,
    size_bin,
    validate_args,
)
from scripts.train_yolo import training_history
from scripts.validate_yolo_dataset import load_metadata


class DatasetMetadataTests(unittest.TestCase):
    def test_yaml_values_must_describe_the_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "conversion_report.json").write_text(
                json.dumps({"class_names": ["brick"], "class_ids": {"brick": 0}}),
                encoding="utf-8",
            )
            (dataset / "classes.txt").write_text("brick\n", encoding="utf-8")
            yaml_path = dataset / "data.yaml"
            valid = "train: images/train\nval: images/val\ntest: images/test\nnames: {0: brick}\n"
            yaml_path.write_text(valid, encoding="utf-8")
            errors: list[str] = []
            load_metadata(dataset, errors)
            self.assertEqual(errors, [])

            yaml_path.write_text(
                valid.replace("val: images/val", "# val: images/val\nval: images/test")
                .replace("names: {0: brick}", "#   0: brick\nnames: [concrete]"),
                encoding="utf-8",
            )
            load_metadata(dataset, errors)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("val to images/val" in error for error in errors))
            self.assertTrue(any("class order" in error for error in errors))


class EvaluationMetricTests(unittest.TestCase):
    def test_in_memory_prediction_receives_bounded_lists(self) -> None:
        class Model:
            def predict(self, *, source, stream, batch):
                self.received_sizes.append(len(source))
                return iter(source)

        model = Model()
        model.received_sizes = []
        paths = [Path(f"{index}.jpg") for index in range(5)]
        self.assertEqual(list(predict_in_batches(model, paths, batch=2)), [str(path) for path in paths])
        self.assertEqual(model.received_sizes, [2, 2, 1])

    def test_duplicate_label_rows_are_reported_without_editing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            label_dir = Path(temporary)
            label_file = label_dir / "sample.txt"
            source = "3 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n"
            label_file.write_text(source, encoding="utf-8")

            duplicates = _duplicate_label_rows(label_dir, ["brick", "concrete", "foam", "general_w"])

            self.assertEqual(duplicates, [{"image": "sample.jpg", "class": "general_w", "duplicate_rows": 1}])
            self.assertEqual(label_file.read_text(encoding="utf-8"), source)

    def test_iou_matrix_handles_overlap_and_empty_inputs(self) -> None:
        iou = box_iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[5, 5, 15, 15]]))
        self.assertAlmostEqual(float(iou[0, 0]), 25 / 175)
        self.assertEqual(box_iou_matrix(np.empty((0, 4)), np.ones((1, 4))).shape, (0, 1))

    def test_duplicate_predictions_can_match_ground_truth_only_once(self) -> None:
        result = match_image(
            gt_boxes=np.array([[0, 0, 10, 10]]),
            gt_classes=np.array([0]),
            pred_boxes=np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
            pred_classes=np.array([0, 0]),
            pred_confidences=np.array([0.9, 0.8]),
        )
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["errors"][0]["type"], "false_positive")

    def test_wrong_class_overlap_is_classification_error(self) -> None:
        result = match_image(
            np.array([[0, 0, 10, 10]]), np.array([0]),
            np.array([[0, 0, 10, 10]]), np.array([1]), np.array([0.9]),
        )
        self.assertEqual(result["errors"][0]["type"], "classification_error")
        self.assertEqual(result["true_positive_gt"], set())

    def test_confusion_counts_keep_one_prediction_per_ground_truth(self) -> None:
        matching = match_image(
            np.array([[0, 0, 10, 10]]), np.array([0]),
            np.array([[0, 0, 10, 10], [0, 0, 10, 10]]), np.array([0, 0]), np.array([0.9, 0.8]),
        )
        record = {
            "gt_classes": np.array([0]),
            "pred_classes": np.array([0, 0]),
            "matching": matching,
        }
        matrix, counts = _make_confusion_matrix(["brick"], [record])
        self.assertEqual(int(matrix.sum()), 2)
        self.assertEqual(counts["class_counts"]["brick"], {"tp": 1, "fp": 1, "fn": 0})

    def test_predefined_size_density_and_overlap_bins(self) -> None:
        self.assertEqual(size_bin([0, 0, 10, 10], 100, 100), "small")
        self.assertEqual(size_bin([0, 0, 30, 30], 100, 100), "medium")
        self.assertEqual(size_bin([0, 0, 40, 40], 100, 100), "large")
        self.assertEqual(density_bin(5), "sparse_1_5")
        self.assertEqual(density_bin(6), "medium_6_9")
        self.assertEqual(density_bin(10), "dense_10_plus")
        bins = overlap_bins(np.array([[0, 0, 10, 10], [8, 0, 18, 10], [40, 40, 50, 50]]))
        self.assertEqual(bins, ["moderate_0_1_0_5", "moderate_0_1_0_5", "low_lt_0_1"])

    def test_fixed_operating_point_f1(self) -> None:
        self.assertAlmostEqual(f1_score(0.5, 1.0), 2 / 3)
        self.assertIsNone(f1_score(None, 1.0))

    def test_reserved_test_requires_explicit_acknowledgement(self) -> None:
        from argparse import Namespace

        args = Namespace(
            split="test", confirm_test=False, checkpoint=Path("missing.pt"), imgsz=None, batch=None,
            conf=0.25, match_iou=0.5, nms_iou=0.7, max_examples=4,
        )
        with self.assertRaisesRegex(SystemExit, "test split is reserved"):
            validate_args(args)


class ExperimentComparisonTests(unittest.TestCase):
    def test_declared_test_run_is_excluded_even_when_it_is_the_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "codd-yolov8n"
            run.mkdir(parents=True)
            (run / "experiment_summary.json").write_text(json.dumps({
                "status": "completed", "training": {"validation_split": "test", "test_split_used": True},
                "validation_metrics": {"metrics/mAP50-95(B)": 0.99},
            }), encoding="utf-8")
            (run / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n1,0.99\n", encoding="utf-8")
            with patch("scripts.compare_experiments._plot_learning_curves") as plot:
                output = compare(run.parent, root / "comparison")
            self.assertEqual(plot.call_args.args[0], [])
            summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["ranked_experiments"], [])
            self.assertTrue(summary["experiments"][0]["test_split_used"])
            self.assertTrue(any("Excluded from comparison" in warning for warning in summary["warnings"]))

    def test_best_epoch_uses_csv_epoch_number_and_first_tie(self) -> None:
        rows = [
            {"epoch": "1", "metrics/mAP50-95(B)": "0.4"},
            {"epoch": "2", "metrics/mAP50-95(B)": "0.7"},
            {"epoch": "3", "metrics/mAP50-95(B)": "0.7"},
        ]
        self.assertEqual(best_epoch_from_results(rows), (2, 0.7))
        zero_based = [{"epoch": "0", "metrics/mAP50-95(B)": "0.2"}, {"epoch": "1", "metrics/mAP50-95(B)": "0.5"}]
        self.assertEqual(best_epoch_from_results(zero_based), (2, 0.5))

    def test_f1_keeps_missing_metrics_missing(self) -> None:
        self.assertAlmostEqual(calculate_f1(0.75, 0.5), 0.6)
        self.assertIsNone(calculate_f1(0.75, None))

    def test_incomplete_run_is_discovered_but_not_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "codd-yolov8n-img640-seed42"
            run_dir.mkdir()
            (run_dir / "args.yaml").write_text("model: yolov8n.pt\nepochs: 50\nimgsz: 640\n", encoding="utf-8")
            experiments = discover_experiments(Path(temporary))
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0].status, "incomplete")
            self.assertFalse(experiments[0].record["eligible_for_ranking"])
            self.assertTrue(any("experiment_summary.json is missing" in warning for warning in experiments[0].warnings))

    def test_loader_reports_epoch_mismatch_and_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "codd-yolov8n-img640-seed42"
            run_dir.mkdir()
            summary = {
                "status": "completed",
                "model": "yolov8n",
                "dataset": "yolo_dataset",
                "training": {"epochs_max": 3, "epochs_executed": 3, "best_epoch": 3, "best_epoch_metric_value": 0.8, "seed": 42},
                "configuration": {"imgsz": 640, "batch": 2, "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "amp": True,
                                  "data": r"C:\Users\synthetic\data.yaml"},
                "validation_metrics": {"metrics/precision(B)": 0.8, "metrics/recall(B)": 0.6, "metrics/mAP50(B)": 0.9, "metrics/mAP50-95(B)": 0.8},
                "validation_per_class": [{"Class": "brick", "mAP50": 0.9, "mAP50-95": 0.8, "Instances": 5}],
            }
            (run_dir / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "args.yaml").write_text("epochs: 3\nimgsz: 640\nbatch: 2\n", encoding="utf-8")
            with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)"])
                writer.writeheader()
                writer.writerows([
                    {"epoch": 1, "metrics/mAP50-95(B)": 0.3, "metrics/mAP50(B)": 0.5},
                    {"epoch": 2, "metrics/mAP50-95(B)": 0.8, "metrics/mAP50(B)": 0.9},
                    {"epoch": 3, "metrics/mAP50-95(B)": 0.7, "metrics/mAP50(B)": 0.8},
                ])
            experiment = load_experiment(run_dir)
            self.assertEqual(experiment.record["best_epoch_csv"], 2)
            self.assertFalse(experiment.record["best_epoch_consistent"])
            self.assertTrue(any("best epoch mismatch" in warning for warning in experiment.warnings))
            self.assertTrue(any("best.pt checkpoint is missing" in warning for warning in experiment.warnings))
            self.assertAlmostEqual(experiment.record["f1"], 2 * 0.8 * 0.6 / (0.8 + 0.6))
            self.assertTrue(experiment.smoke_test)
            self.assertTrue(experiment.record["eligible_for_ranking"])
            self.assertEqual(experiment.record["configuration"]["data"], "data.yaml")
            self.assertNotIn("args", experiment.record)

    def test_comparator_generates_artifacts_for_one_synthetic_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            run_dir = runs / "codd-yolov8n-img640-seed17"
            run_dir.mkdir(parents=True)
            (run_dir / "weights").mkdir()
            (run_dir / "weights" / "best.pt").write_bytes(b"synthetic checkpoint placeholder")
            summary = {
                "status": "completed", "model": "yolov8n", "dataset": "synthetic",
                "training": {"epochs_max": 4, "epochs_executed": 3, "best_epoch": 2, "best_epoch_metric_value": 0.8,
                             "seed": 17, "training_seconds": 12.0, "validation_seconds": 1.0,
                             "validation_split": "val", "test_split_used": False},
                "configuration": {"imgsz": 640, "batch": 2, "optimizer": "AdamW", "lr0": 0.001,
                                  "weight_decay": 0.0005, "amp": True},
                "validation_metrics": {"metrics/precision(B)": 0.8, "metrics/recall(B)": 0.6,
                                       "metrics/mAP50(B)": 0.9, "metrics/mAP50-95(B)": 0.8},
                "validation_per_class": [{"Class": "brick", "Instances": 3, "mAP50": 0.9,
                                          "mAP50-95": 0.8, "Box-P": 0.8, "Box-R": 0.6, "Box-F1": 0.69}],
                "inference_latency_ms_per_image": 8.0, "inference_images_per_second": 125.0,
                "hardware": {"system": "synthetic"}, "software": {"ultralytics": "8.4.210", "torch": "2.x"},
            }
            (run_dir / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "args.yaml").write_text("epochs: 4\nimgsz: 640\nbatch: 2\n", encoding="utf-8")
            with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "metrics/mAP50-95(B)", "train/box_loss", "val/box_loss"])
                writer.writeheader()
                writer.writerows([
                    {"epoch": 1, "metrics/mAP50-95(B)": 0.5, "train/box_loss": 1.2, "val/box_loss": 1.1},
                    {"epoch": 2, "metrics/mAP50-95(B)": 0.8, "train/box_loss": 0.9, "val/box_loss": 0.8},
                    {"epoch": 3, "metrics/mAP50-95(B)": 0.7, "train/box_loss": 0.8, "val/box_loss": 0.9},
                ])
            output = root / "comparison"
            compare(runs, output)
            for filename in (
                "comparison.csv", "comparison.json", "report.md", "global_metrics.png", "learning_curves.png",
                "loss_curves.png", "per_class_ap.png", "class_model_heatmap.png", "training_time.png", "map_latency.png",
            ):
                self.assertTrue((output / filename).is_file(), filename)


class TrainingHistoryTests(unittest.TestCase):
    def test_best_epoch_respects_one_based_ultralytics_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results.csv"
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "metrics/mAP50-95(B)"])
                writer.writeheader()
                writer.writerows([
                    {"epoch": 1, "metrics/mAP50-95(B)": 0.3},
                    {"epoch": 2, "metrics/mAP50-95(B)": 0.8},
                ])

            history = training_history(results, trainer=None)

        self.assertEqual(history["best_epoch"], 2)
        self.assertEqual(history["epochs_executed"], 2)

    def test_best_epoch_converts_zero_based_csv_to_one_based_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results.csv"
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "metrics/mAP50-95(B)"])
                writer.writeheader()
                writer.writerows([
                    {"epoch": 0, "metrics/mAP50-95(B)": 0.3},
                    {"epoch": 1, "metrics/mAP50-95(B)": 0.8},
                ])

            history = training_history(results, trainer=None)

        self.assertEqual(history["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
