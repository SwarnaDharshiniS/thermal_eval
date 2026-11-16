"""Compare YOLOv8 and Faster R-CNN, with optional SSD, on the same COCO test split.

COCO mAP uses score >= 0.001 and at most 100 detections per image. Micro
precision/recall use score >= 0.25 (configurable) and class-aware IoU >= 0.5.
Latency includes model preprocessing, transfer, forward and postprocessing,
but excludes disk decoding and two warmup iterations. All models use one device.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from detection_common import bbox_iou, categories, crowd_overlap, detection_metrics, image_path, load_rgb, require_valid
from train_torchvision import load_model, predict_dataset, synchronize


def category_signature(category_list):
    return [(category["id"], category["name"]) for category in category_list]


def predict_yolo(checkpoint, mapping_path, data, images_folder, device, imgsz=640):
    from ultralytics import YOLO

    category_list = json.loads(Path(mapping_path).read_text())["categories"]
    if category_signature(category_list) != category_signature(categories(data)):
        raise ValueError("YOLO class mapping does not match the test dataset")
    model = YOLO(str(checkpoint))
    if [model.names[index] for index in range(len(model.names))] != [category["name"] for category in category_list]:
        raise ValueError("YOLO checkpoint class names do not match category_mapping.json")
    yolo_device = str(device) if torch.device(device).type != "cuda" else str(torch.device(device).index or 0)
    def infer(pixels):
        return model.predict(source=np.asarray(pixels)[:, :, ::-1].copy(), imgsz=imgsz, device=yolo_device, conf=0.001, max_det=100, verbose=False)[0]
    first = load_rgb(image_path(images_folder, data["images"][0]["file_name"]))
    for iteration in range(2):
        infer(first)
    predictions, total_time = [], 0.0
    for image in data["images"]:
        pixels = load_rgb(image_path(images_folder, image["file_name"]))
        synchronize(device)
        start = time.perf_counter()
        result = infer(pixels)
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        labels = result.boxes.cls.detach().cpu().tolist()
        synchronize(device)
        total_time += time.perf_counter() - start
        for box, label, score in zip(boxes, labels, scores):
            left, top, right, bottom = box
            if right > left and bottom > top:
                predictions.append({"image_id": image["id"], "category_id": category_list[int(label)]["id"], "bbox": [left, top, right - left, bottom - top], "score": score})
    return predictions, total_time / len(data["images"])


def save_comparison(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plot

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = [row["model"] for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plot.subplots(1, 2, figsize=(12, 5))
    fields = ["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]
    for index, field in enumerate(fields):
        axes[0].bar(positions + (index - 1.5) * .18, [row[field] for row in rows], width=.18, label=field)
    axes[0].set_xticks(positions, names)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Detection accuracy")
    colors = ["#2a9d8f", "#e76f51", "#457b9d"][:len(names)]
    axes[1].bar(names, [row["inference_ms_per_image"] for row in rows], color=colors)
    axes[1].set_ylabel("Milliseconds per image")
    axes[1].set_title("Inference latency")
    figure.tight_layout()
    figure.savefig(output / "results.png", dpi=160)
    plot.close(figure)

    metric_fields = ["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]
    metric_values = np.array([[row[field] for field in metric_fields] for row in rows])
    figure, axis = plot.subplots(figsize=(8, max(3, len(rows) * 1.1)))
    heatmap = axis.imshow(metric_values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(metric_fields)), metric_fields, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(names)), names)
    axis.set_title("Model metric comparison")
    for row_index in range(len(rows)):
        for column_index in range(len(metric_fields)):
            axis.text(column_index, row_index, f"{metric_values[row_index, column_index]:.3f}", ha="center", va="center")
    figure.colorbar(heatmap, ax=axis, label="Score")
    figure.tight_layout()
    figure.savefig(output / "metrics_heatmap.png", dpi=160)
    plot.close(figure)


def detection_confusion_matrix(data, predictions, confidence=0.25, iou_threshold=0.5):
    category_list = categories(data)
    category_index = {category["id"]: index for index, category in enumerate(category_list)}
    background_index = len(category_list)
    matrix = np.zeros((background_index + 1, background_index + 1), dtype=np.int64)
    truths_by_image = defaultdict(list)
    crowds_by_image = defaultdict(list)
    predictions_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        target = crowds_by_image if annotation.get("iscrowd", 0) else truths_by_image
        target[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        if prediction["score"] >= confidence:
            predictions_by_image[prediction["image_id"]].append(prediction)

    for image in data["images"]:
        image_id = image["id"]
        unmatched_truths = {item["id"]: item for item in truths_by_image[image_id]}
        selected = sorted(predictions_by_image[image_id], key=lambda item: item["score"], reverse=True)
        for prediction in selected:
            prediction_index = category_index[prediction["category_id"]]
            best_truth = max(
                unmatched_truths.values(),
                key=lambda item: bbox_iou(prediction["bbox"], item["bbox"]),
                default=None,
            )
            overlap = bbox_iou(prediction["bbox"], best_truth["bbox"]) if best_truth else 0
            if best_truth is not None and overlap >= iou_threshold:
                matrix[category_index[best_truth["category_id"]], prediction_index] += 1
                del unmatched_truths[best_truth["id"]]
            elif any(crowd_overlap(prediction["bbox"], crowd["bbox"]) >= iou_threshold for crowd in crowds_by_image[image_id]):
                continue
            else:
                matrix[background_index, prediction_index] += 1
        for truth in unmatched_truths.values():
            matrix[category_index[truth["category_id"]], background_index] += 1

    labels = [category["name"] for category in category_list] + ["background"]
    return matrix, labels


def save_diagnostics(data, predictions_by_model, output, confidence=0.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plot

    output = Path(output)
    category_list = categories(data)
    class_rows = []
    for model_name, predictions in predictions_by_model.items():
        matrix, labels = detection_confusion_matrix(data, predictions, confidence)
        slug = model_name.lower().replace(" ", "_").replace("-", "_")
        csv_path = output / f"confusion_matrix_{slug}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true\\pred", *labels])
            writer.writerows([[label, *row.tolist()] for label, row in zip(labels, matrix)])

        figure, axis = plot.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
        heatmap = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xlabel("Predicted category")
        axis.set_ylabel("Ground-truth category")
        axis.set_title(f"{model_name} confusion matrix (IoU >= 0.50, score >= {confidence:.2f})")
        cutoff = matrix.max() / 2 if matrix.size else 0
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                value = matrix[row_index, column_index]
                axis.text(column_index, row_index, str(value), ha="center", va="center", color="white" if value > cutoff else "black")
        figure.colorbar(heatmap, ax=axis, label="Detection count")
        figure.tight_layout()
        figure.savefig(output / f"confusion_matrix_{slug}.png", dpi=160)
        plot.close(figure)

        for index, category in enumerate(category_list):
            true_positive = int(matrix[index, index])
            false_positive = int(matrix[:, index].sum() - true_positive)
            false_negative = int(matrix[index, :].sum() - true_positive)
            precision = true_positive / max(1, true_positive + false_positive)
            recall = true_positive / max(1, true_positive + false_negative)
            class_rows.append({
                "model": model_name,
                "category": category["name"],
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(1e-12, precision + recall),
            })

    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--yolo-mapping", type=Path, required=True)
    parser.add_argument("--fasterrcnn", type=Path, required=True)
    parser.add_argument("--ssd", type=Path, help="Optional SSD checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    arguments = parser.parse_args()
    if not 0 <= arguments.confidence <= 1:
        parser.error("Confidence must be between 0 and 1")
    if (arguments.output / "comparison.csv").exists() or (arguments.output / "results.png").exists():
        parser.error("Comparison outputs already exist; choose a new output directory")
    data = require_valid(arguments.annotations, arguments.images)
    if not data["images"]:
        parser.error("Test split is empty")
    device = torch.device(arguments.device)
    rows = []
    predictions_by_model = {}
    predictions, elapsed = predict_yolo(arguments.yolo, arguments.yolo_mapping, data, arguments.images, device, arguments.imgsz)
    predictions_by_model["YOLOv8s"] = predictions
    rows.append(dict(model="YOLOv8s", **detection_metrics(data, predictions, arguments.confidence), inference_ms_per_image=elapsed * 1000))
    models = [("Faster R-CNN", "fasterrcnn", arguments.fasterrcnn)]
    if arguments.ssd is not None:
        models.append(("SSD300", "ssd", arguments.ssd))
    for label, architecture, checkpoint in models:
        model, category_list, saved_architecture = load_model(checkpoint, device)
        if saved_architecture != architecture or category_signature(category_list) != category_signature(categories(data)):
            raise ValueError(f"{label}: checkpoint architecture/categories do not match test dataset")
        predictions, elapsed = predict_dataset(model, data, arguments.images, category_list, device)
        predictions_by_model[label] = predictions
        rows.append(dict(model=label, **detection_metrics(data, predictions, arguments.confidence), inference_ms_per_image=elapsed * 1000))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_comparison(rows, arguments.output)
    save_diagnostics(data, predictions_by_model, arguments.output, arguments.confidence)
    print(json.dumps(rows, indent=2))
    print(f"Saved comparison charts, confusion matrices, and per-class metrics to {arguments.output}")


if __name__ == "__main__":
    main()