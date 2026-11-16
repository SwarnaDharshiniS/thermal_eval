# RGB and Thermal Object Detection

Run from the workspace root using `.venv/Scripts/python.exe`. Dependencies are
listed in `requirements_detection.txt`. The current virtual environment has CPU
PyTorch; `cuda:0` requires a CUDA-enabled PyTorch installation and a compatible GPU.
Full training is expensive on CPU. Pretrained model downloads require internet.

## Separate Workflows

All commands below process both `RGB_Dataset` and `Thermal_Dataset` independently.
Use `--datasets RGB_Dataset` or `--datasets Thermal_Dataset` for just one modality.

```powershell
# Report source integrity (RGB currently has seven 0.01-pixel box overshoots).
& .venv/Scripts/python.exe run_detection_pipeline.py --stage validate

# Prepare once: repaired JSON copies, 3x training images, and YOLO exports.
& .venv/Scripts/python.exe run_detection_pipeline.py --stage prepare --clip-boxes

# Full training: YOLOv8s 50 epochs at 640; Faster R-CNN 20/batch 4; SSD 20/batch 8.
& .venv/Scripts/python.exe run_detection_pipeline.py --stage train --device cpu

# Compare the three best checkpoints on each modality's unchanged test images.
& .venv/Scripts/python.exe run_detection_pipeline.py --stage evaluate --device cpu
```

For a fresh output, `--stage all --clip-boxes` performs preparation, training,
and evaluation. After preparation has already run, use the separate train and
evaluate stages, not `all`. Existing preparation/checkpoint/result files are not
silently overwritten. Source annotations are never changed. Generated training
images (`aug_<image_id>_<variant>.png`) are written into the same training folder.
The augmented JSON includes originals plus two variants, not three extra variants.
Validation and test images are never augmented.

Outputs for each modality:

```text
detection_runs/<RGB_Dataset or Thermal_Dataset>/
  validation_train.json, validation_val.json, validation_test.json
  instances_train.json, instances_val.json, instances_test.json
  instances_train_augmented.json
  yolo/images/{train,val,test}/
  yolo/labels/{train,val,test}/
  yolo/data.yaml
  yolo/category_mapping.json
  models/yolov8s/weights/best.pt
  models/fasterrcnn/best.pt
  models/ssd/best.pt
  results/comparison.csv
  results/results.png
  results/metrics_heatmap.png
  results/per_class_metrics.csv
  results/confusion_matrix_<model>.csv
  results/confusion_matrix_<model>.png
```

Confusion matrices use score >= 0.25 and class-aware IoU >= 0.5. Rows are
ground-truth categories, columns are predicted categories, and the background
row/column counts false positives and missed detections. Per-class precision,
recall and F1 use those same matched detections.

## Individual Scripts

```powershell
& .venv/Scripts/python.exe validate_coco.py --annotations Thermal_Dataset/annotations/instances_train.json --images Thermal_Dataset/images/train --report thermal_validation.json
& .venv/Scripts/python.exe augment_coco.py --annotations Thermal_Dataset/annotations/instances_train.json --images Thermal_Dataset/images/train --output thermal_augmented.json --factor 3 --seed 42
& .venv/Scripts/python.exe coco_to_yolo.py --dataset Thermal_Dataset --train-annotations thermal_augmented.json --output thermal_yolo --train --epochs 50 --imgsz 640 --device cpu
& .venv/Scripts/python.exe train_torchvision.py --dataset Thermal_Dataset --train-annotations thermal_augmented.json --model fasterrcnn --epochs 20 --batch 4 --output thermal_fasterrcnn
& .venv/Scripts/python.exe train_torchvision.py --dataset Thermal_Dataset --train-annotations thermal_augmented.json --model ssd --epochs 20 --batch 8 --output thermal_ssd
& .venv/Scripts/python.exe evaluate_models.py --annotations Thermal_Dataset/annotations/instances_test.json --images Thermal_Dataset/images/test --yolo runs/detection/yolov8s/weights/best.pt --yolo-mapping thermal_yolo/category_mapping.json --fasterrcnn thermal_fasterrcnn/best.pt --ssd thermal_ssd/best.pt --output thermal_results
```

These are alternatives to the runner, not commands to run on top of already
generated augmentation files. Use each script's `--help` for options. RGB's source
JSON needs clipping before use; `validate_coco.py --clip-boxes-output NEW_JSON`
creates a repaired copy. Missing/corrupt images, invalid IDs/references, and other
errors must be corrected separately; clipping does not hide them.

## Data and Metric Contract

- EXIF orientation is applied before dimensions, boxes, augmentation, conversion,
  training and evaluation are interpreted. YOLO-exported PNGs have normalized orientation.
- Category IDs may be noncontiguous. YOLO uses zero-based indices; torchvision
  uses one-based labels with zero reserved for background. Checkpoints/mapping
  files retain the exact original COCO categories. Synonyms are not auto-merged.
- Training drops crowd annotations consistently. Test evaluation retains COCO
  crowd handling. The workflow is box detection only; augmentation rejects
  nonempty segmentation/keypoint annotations rather than leaving them stale.
- Augmentations: horizontal flip, brightness/contrast, rotation within +/-15
  degrees, Gaussian blur and hue/saturation/value. Transformed boxes are clipped
  and dropped below 10% visibility; images are still kept, including backgrounds.
  HSV shifts follow the request for both modalities; they may be inappropriate
  for radiometric thermal data and should be assessed before scientific use.
- Faster R-CNN uses pretrained ResNet50-FPN; SSD300 uses pretrained VGG16.
  Both replace the class head, train all unfrozen parameters, and save `best.pt`
  by validation COCO mAP@0.5:0.95. YOLO saves Ultralytics' native best checkpoint.
- Final metrics for all three models use the exact same COCO test JSON per
  modality via pycocotools. mAP is reported at IoU 0.5 and over 0.5:0.95.
  Scores are fractions, not percentages. AP prediction threshold is 0.001 with
  at most 100 detections/image. Precision and recall are micro-averaged,
  class-aware, one-to-one matches at IoU 0.5 and confidence 0.25.
- Inference time is milliseconds/image at batch 1, including model-specific
  preprocessing, device transfer, forward pass and postprocessing; image disk
  decoding and two warmup iterations are excluded. CUDA is synchronized.
- Original random image splits are retained. No scene grouping or duplicate
  removal is performed: check related frames/duplicate images across splits
  before interpreting test results as an unbiased generalization estimate.

## Tests

```powershell
& .venv/Scripts/python.exe -m unittest test_detection_pipeline -v
```

Tests use tiny synthetic images and one epoch, without pretrained downloads.
They are smoke/regression checks, not trained model results or accuracy claims.