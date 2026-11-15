# Model evaluation report

- Model: `yolov8n.pt`
- Split: `test`
- Images: 573
- Raw label objects: 3224
- Ground-truth objects scored by Ultralytics: 3224
- Checkpoint: `runs/detect/codd-yolov8n-img640-seed42/weights/best.pt`
- Input size / batch: 640 / 2

## Global metrics

AP metrics come from Ultralytics' native validator. Precision, recall, and F1 are computed from the diagnostic one-to-one matches at the fixed operating point below.

- mAP50-95: 0.7441794866822836
- mAP50: 0.8027755088071216
- Precision: 0.9287429943955164
- Recall: 0.7196029776674938
- F1: 0.8109052778748689
- Operating point: confidence >= 0.5, one-to-one IoU >= 0.5; NMS IoU=0.5.

## Per-class results

| Class | Instances | AP50 | AP50-95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| brick | 351 | 0.98021 | 0.93878 | 0.9792284866468842 | 0.9401709401709402 | 0.9593023255813953 |
| concrete | 384 | 0.95383 | 0.91462 | 0.9445910290237467 | 0.9322916666666666 | 0.9384010484927915 |
| foam | 300 | 0.80465 | 0.75422 | 0.8724279835390947 | 0.7066666666666667 | 0.7808471454880295 |
| general_w | 266 | 0.75595 | 0.70133 | 0.864321608040201 | 0.6466165413533834 | 0.7397849462365591 |
| gypsum_board | 317 | 0.88226 | 0.82848 | 0.9616724738675958 | 0.8706624605678234 | 0.9139072847682119 |
| pipes | 256 | 0.68776 | 0.5737 | 0.9139784946236559 | 0.33203125 | 0.48710601719197705 |
| plastic | 266 | 0.52736 | 0.45163 | 0.7101449275362319 | 0.3684210526315789 | 0.4851485148514851 |
| stone | 270 | 0.66562 | 0.62456 | 0.993103448275862 | 0.5333333333333333 | 0.6939759036144578 |
| tile | 375 | 0.92978 | 0.89264 | 0.9512893982808023 | 0.8853333333333333 | 0.9171270718232043 |
| wood | 439 | 0.84033 | 0.76183 | 0.9542682926829268 | 0.7129840546697038 | 0.816166883963494 |

## Performance by object and scene

Recall by size, scene density, and overlap uses ground-truth subsets at the same fixed confidence/IoU operating point. AP is not recomputed on subsets.

- Size rule: small <= 1% image area; medium > 1% and <= 10%; large > 10%. These fixed bins reflect the dataset audit, which found about half of boxes below 1% area.
- Density rule: sparse 1-5 objects, medium 6-9, dense >= 10; cutoffs reflect the audit's concentration around <=5 and >=10 objects per image.
- Overlap rule: maximum pairwise GT IoU < 0.1, 0.1-<0.5, or >=0.5. Single-object scenes fall in the low-overlap group.

- Size recall: `{'small': {'ground_truth_objects': 1609, 'true_positives': 1157, 'recall': 0.7190801740211311}, 'medium': {'ground_truth_objects': 1513, 'true_positives': 1081, 'recall': 0.7144745538664904}, 'large': {'ground_truth_objects': 102, 'true_positives': 82, 'recall': 0.803921568627451}}`
- Density recall: `{'sparse_1_5': {'ground_truth_objects': 766, 'true_positives': 644, 'recall': 0.8407310704960835}, 'medium_6_9': {'ground_truth_objects': 351, 'true_positives': 274, 'recall': 0.7806267806267806}, 'dense_10_plus': {'ground_truth_objects': 2107, 'true_positives': 1402, 'recall': 0.6654010441385857}}`
- Overlap recall: `{'low_lt_0_1': {'ground_truth_objects': 2680, 'true_positives': 2223, 'recall': 0.8294776119402985}, 'moderate_0_1_0_5': {'ground_truth_objects': 518, 'true_positives': 97, 'recall': 0.18725868725868725}, 'high_ge_0_5': {'ground_truth_objects': 26, 'true_positives': 0, 'recall': 0.0}}`

## Errors and plots

- Object-level errors: `{'classification_error': 144, 'false_negative': 745, 'false_positive': 19, 'localization_error': 15}`
- Most frequent cross-class confusions: `[{'ground_truth_class': 'plastic', 'predicted_class': 'general_w', 'count': 15}, {'ground_truth_class': 'tile', 'predicted_class': 'concrete', 'count': 12}, {'ground_truth_class': 'foam', 'predicted_class': 'gypsum_board', 'count': 11}, {'ground_truth_class': 'brick', 'predicted_class': 'tile', 'count': 9}, {'ground_truth_class': 'gypsum_board', 'predicted_class': 'foam', 'count': 9}, {'ground_truth_class': 'foam', 'predicted_class': 'plastic', 'count': 9}, {'ground_truth_class': 'stone', 'predicted_class': 'plastic', 'count': 8}, {'ground_truth_class': 'plastic', 'predicted_class': 'pipes', 'count': 8}, {'ground_truth_class': 'wood', 'predicted_class': 'foam', 'count': 7}, {'ground_truth_class': 'general_w', 'predicted_class': 'wood', 'count': 7}]`
- Error rows: `error_analysis.csv`
- Confusion matrices: `confusion_matrix_absolute.png`, `confusion_matrix_normalized.png`
- Class metrics: `per_class_metrics.png`
- Group recall: `recall_by_size.png`, `recall_by_density.png`, `recall_by_overlap.png`
- Native Ultralytics PR curves and validation plots: `ultralytics/native_validation/`
- GT/prediction side-by-side examples: `visual_examples/`

## Runtime and limitations

- Native validation wall time: 29.276 seconds.
- Error-analysis prediction pass wall time: 89.415 seconds; this includes batching, preprocessing, inference, postprocessing, and result transfer.
- Native per-image speed breakdown: `{'preprocess': 0.6510439799386638, 'inference': 9.34977731285503, 'postprocess': 1.9738349027811413}` ms/image.
- Parameters: 3140190
- One-to-one matching is confidence-ordered greedy matching. It is used only for thresholded diagnostics; AP remains Ultralytics-native.
- Group recall is descriptive and depends on the specified confidence, IoU, and fixed bins; it is not a replacement for mAP.
