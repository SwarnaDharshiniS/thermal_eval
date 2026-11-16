"""Compare YOLOv8 and Faster R-CNN, with optional SSD, on the same COCO test split.

COCO mAP uses score >= 0.001 and at most 100 detections per image. Micro
precision/recall use score >= 0.25 (configurable) and class-aware IoU >= 0.5.
Latency includes model preprocessing, transfer, forward and postprocessing,
but excludes disk decoding and two warmup iterations. All models use one device.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from detection_common import categories, detection_metrics, image_path, load_rgb, require_valid
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
    predictions, elapsed = predict_yolo(arguments.yolo, arguments.yolo_mapping, data, arguments.images, device, arguments.imgsz)
    rows.append(dict(model="YOLOv8s", **detection_metrics(data, predictions, arguments.confidence), inference_ms_per_image=elapsed * 1000))
    models = [("Faster R-CNN", "fasterrcnn", arguments.fasterrcnn)]
    if arguments.ssd is not None:
        models.append(("SSD300", "ssd", arguments.ssd))
    for label, architecture, checkpoint in models:
        model, category_list, saved_architecture = load_model(checkpoint, device)
        if saved_architecture != architecture or category_signature(category_list) != category_signature(categories(data)):
            raise ValueError(f"{label}: checkpoint architecture/categories do not match test dataset")
        predictions, elapsed = predict_dataset(model, data, arguments.images, category_list, device)
        rows.append(dict(model=label, **detection_metrics(data, predictions, arguments.confidence), inference_ms_per_image=elapsed * 1000))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_comparison(rows, arguments.output)
    print(json.dumps(rows, indent=2))
    print(f"Saved {arguments.output / 'comparison.csv'} and {arguments.output / 'results.png'}")


if __name__ == "__main__":
    main()