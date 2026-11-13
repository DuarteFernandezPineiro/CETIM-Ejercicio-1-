# CODD dataset audit

- Dataset: `Construction and Demolition Waste Object Detection Dataset  (CODD)`
- Images: **3,127**
- Annotated objects: **16,519**
- Classes: **10** (brick, concrete, foam, general_w, gypsum_board, pipes, plastic, stone, tile, wood)
- Exact duplicate groups across splits: **0**
- Basenames reused across splits: **21** (keep split folders separate)
- Data-quality issues recorded: **42**

## Split summary

| Split | Images | XML | Objects | Mean objects/image | Dimensions |
|---|---:|---:|---:|---:|---|
| training | 1,984 | 1,984 | 10,058 | 5.07 | 1920x1200: 1984 |
| validation | 570 | 570 | 3,237 | 5.68 | 1920x1200: 570 |
| testing | 573 | 573 | 3,224 | 5.63 | 1920x1200: 573 |

## Class counts

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| brick | 1,370 | 372 | 351 |
| concrete | 1,407 | 364 | 384 |
| foam | 746 | 295 | 300 |
| general_w | 742 | 288 | 266 |
| gypsum_board | 1,184 | 276 | 317 |
| pipes | 715 | 260 | 256 |
| plastic | 675 | 306 | 266 |
| stone | 744 | 245 | 270 |
| tile | 1,344 | 346 | 375 |
| wood | 1,131 | 485 | 439 |

## Figures

- `class_distribution.png`: object instances and class balance by split.
- `objects_per_image.png`: image density by split.
- `bbox_geometry.png`: relative box sizes and area distribution.
- `annotated_examples.png`: visual spot-check of example boxes.

## Data quality

See `data_issues.csv` for per-file issues. **42 / 16,519 boxes (0.25%) extend beyond image bounds.** Review these border boxes when converting from PASCAL VOC coordinates to the training framework’s convention.

## Evaluation guidance

For object detection, report mAP@0.5 and mAP@0.5:0.95 plus per-class precision/recall. Avoid accuracy as the headline metric. Keep `testing` untouched until the final evaluation; use `validation` for model selection.
