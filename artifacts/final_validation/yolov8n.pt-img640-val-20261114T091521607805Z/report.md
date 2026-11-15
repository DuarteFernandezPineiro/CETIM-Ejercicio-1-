# Model evaluation report

- Model: `yolov8n.pt`
- Split: `val`
- Images: 570
- Raw label objects: 3237
- Ground-truth objects scored by Ultralytics: 3236
- Checkpoint: `runs/detect/codd-yolov8n-img640-seed42/weights/best.pt`
- Input size / batch: 640 / 2

## Global metrics

AP metrics come from Ultralytics' native validator. Precision, recall, and F1 are computed from the diagnostic one-to-one matches at the fixed operating point below.

- mAP50-95: 0.7209145055254345
- mAP50: 0.7853985664289954
- Precision: 0.9284818067754078
- Recall: 0.6858202038924931
- F1: 0.7889125799573561
- Operating point: confidence >= 0.5, one-to-one IoU >= 0.5; NMS IoU=0.5.

## Per-class results

| Class | Instances | AP50 | AP50-95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| brick | 372 | 0.95137 | 0.91389 | 0.9767441860465116 | 0.9032258064516129 | 0.9385474860335195 |
| concrete | 364 | 0.96711 | 0.92687 | 0.9607843137254902 | 0.9423076923076923 | 0.9514563106796117 |
| foam | 295 | 0.74739 | 0.65733 | 0.8549222797927462 | 0.559322033898305 | 0.6762295081967212 |
| general_w | 287 | 0.75105 | 0.68363 | 0.8779342723004695 | 0.6493055555555556 | 0.7465069860279441 |
| gypsum_board | 276 | 0.90294 | 0.85543 | 1.0 | 0.8876811594202898 | 0.9404990403071017 |
| pipes | 260 | 0.55674 | 0.45198 | 0.9253731343283582 | 0.23846153846153847 | 0.3792048929663609 |
| plastic | 306 | 0.5158 | 0.4253 | 0.6742857142857143 | 0.38562091503267976 | 0.4906444906444906 |
| stone | 245 | 0.66633 | 0.61491 | 0.9914529914529915 | 0.47346938775510206 | 0.6408839779005526 |
| tile | 346 | 0.94307 | 0.91056 | 0.9319526627218935 | 0.9104046242774566 | 0.9210526315789473 |
| wood | 485 | 0.85221 | 0.76925 | 0.9736842105263158 | 0.6865979381443299 | 0.8053204353083434 |

## Performance by object and scene

Recall by size, scene density, and overlap uses ground-truth subsets at the same fixed confidence/IoU operating point. AP is not recomputed on subsets.

- Size rule: small <= 1% image area; medium > 1% and <= 10%; large > 10%. These fixed bins reflect the dataset audit, which found about half of boxes below 1% area.
- Density rule: sparse 1-5 objects, medium 6-9, dense >= 10; cutoffs reflect the audit's concentration around <=5 and >=10 objects per image.
- Overlap rule: maximum pairwise GT IoU < 0.1, 0.1-<0.5, or >=0.5. Single-object scenes fall in the low-overlap group.

- Size recall: `{'small': {'ground_truth_objects': 1498, 'true_positives': 1044, 'recall': 0.6969292389853138}, 'medium': {'ground_truth_objects': 1640, 'true_positives': 1099, 'recall': 0.6701219512195122}, 'large': {'ground_truth_objects': 99, 'true_positives': 77, 'recall': 0.7777777777777778}}`
- Density recall: `{'sparse_1_5': {'ground_truth_objects': 758, 'true_positives': 656, 'recall': 0.8654353562005277}, 'medium_6_9': {'ground_truth_objects': 385, 'true_positives': 287, 'recall': 0.7454545454545455}, 'dense_10_plus': {'ground_truth_objects': 2094, 'true_positives': 1277, 'recall': 0.6098376313276027}}`
- Overlap recall: `{'low_lt_0_1': {'ground_truth_objects': 2583, 'true_positives': 2125, 'recall': 0.8226867982965544}, 'moderate_0_1_0_5': {'ground_truth_objects': 612, 'true_positives': 92, 'recall': 0.1503267973856209}, 'high_ge_0_5': {'ground_truth_objects': 42, 'true_positives': 3, 'recall': 0.07142857142857142}}`

## Errors and plots

- Object-level errors: `{'classification_error': 127, 'false_positive': 23, 'false_negative': 869, 'localization_error': 21}`
- Most frequent cross-class confusions: `[{'ground_truth_class': 'foam', 'predicted_class': 'plastic', 'count': 19}, {'ground_truth_class': 'brick', 'predicted_class': 'tile', 'count': 14}, {'ground_truth_class': 'general_w', 'predicted_class': 'plastic', 'count': 13}, {'ground_truth_class': 'plastic', 'predicted_class': 'general_w', 'count': 13}, {'ground_truth_class': 'tile', 'predicted_class': 'concrete', 'count': 8}, {'ground_truth_class': 'pipes', 'predicted_class': 'plastic', 'count': 8}, {'ground_truth_class': 'plastic', 'predicted_class': 'foam', 'count': 5}, {'ground_truth_class': 'brick', 'predicted_class': 'concrete', 'count': 4}, {'ground_truth_class': 'concrete', 'predicted_class': 'tile', 'count': 4}, {'ground_truth_class': 'wood', 'predicted_class': 'foam', 'count': 4}]`
- Error rows: `error_analysis.csv`
- Confusion matrices: `confusion_matrix_absolute.png`, `confusion_matrix_normalized.png`
- Class metrics: `per_class_metrics.png`
- Group recall: `recall_by_size.png`, `recall_by_density.png`, `recall_by_overlap.png`
- Native Ultralytics PR curves and validation plots: `ultralytics/native_validation/`
- GT/prediction side-by-side examples: `visual_examples/`

## Runtime and limitations

- Native validation wall time: 26.55 seconds.
- Error-analysis prediction pass wall time: 97.792 seconds; this includes batching, preprocessing, inference, postprocessing, and result transfer.
- Native per-image speed breakdown: `{'preprocess': 0.701158069964629, 'inference': 8.159127368146487, 'postprocess': 2.268555789283199}` ms/image.
- Parameters: 3140190
- One-to-one matching is confidence-ordered greedy matching. It is used only for thresholded diagnostics; AP remains Ultralytics-native.
- Group recall is descriptive and depends on the specified confidence, IoU, and fixed bins; it is not a replacement for mAP.
- Label-cache discrepancy: Ultralytics scored 3236 objects after removing duplicate label rows; raw files contain 3237.
- Affected classes: general_w: native=287, raw=288.

- Exact duplicate YOLO label rows (reported, not edited): `[{'image': 'gen-1029.jpg', 'class': 'general_w', 'duplicate_rows': 1}]`

