# YOLO experiment comparison

Only validation metrics are used for selection. The reserved test split is never loaded by this comparator.

## Ranking by validation mAP50-95

| Rank | Model | mAP50-95 | mAP50 | Precision | Recall | F1 | Epochs | Best epoch (CSV) | Train h | Latency ms/img |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | yolov8n | 0.7209 | 0.7854 | 0.8500 | 0.7067 | 0.7718 | 60 | 55 | 1.51 | 6.090 |
| 2 | yolo11n | 0.6958 | 0.7676 | 0.8420 | 0.7057 | 0.7678 | 45 | 35 | 1.23 | 6.498 |

## Per-class AP50-95

| Class | yolov8n | yolo11n |
|---|---:|---:|
| brick | 0.9139 | 0.8915 |
| concrete | 0.9269 | 0.9048 |
| foam | 0.6573 | 0.6551 |
| general_w | 0.6836 | 0.6869 |
| gypsum_board | 0.8554 | 0.8223 |
| pipes | 0.4520 | 0.4279 |
| plastic | 0.4253 | 0.4158 |
| stone | 0.6149 | 0.4779 |
| tile | 0.9106 | 0.8876 |
| wood | 0.7692 | 0.7878 |

## Comparability warnings

No differences were detected in the recorded comparability fields among eligible runs.

## Interpretation notes

- F1 is derived as 2 × precision × recall / (precision + recall) using the recorded validation operating point. It is not AP.
- The best epoch is read from `results.csv` using the experiment's recorded checkpoint-selection metric (mAP50-95 for this detection task). Summary/CSV mismatches are reported rather than silently hidden.
- Training curves and loss plots retain each experiment's original epoch log. A single completed experiment is valid input and still produces the plots.
- Runtime and hardware differences can make wall time and latency comparisons non-equivalent.

## Figures

- `global_metrics.png`
- `learning_curves.png`
- `loss_curves.png`
- `per_class_ap.png` and `class_model_heatmap.png`
- `training_time.png`
- `map_latency.png`
