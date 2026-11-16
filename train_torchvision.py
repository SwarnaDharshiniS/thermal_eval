"""Train pretrained Faster R-CNN ResNet50-FPN or SSD300-VGG16 on COCO splits."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn, ssd300_vgg16,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.transforms.functional import pil_to_tensor

from detection_common import categories, detection_metrics, image_path, load_rgb, require_valid


class CocoDetectionDataset(Dataset):
    def __init__(self, annotations, images_folder):
        self.data = require_valid(annotations, images_folder)
        self.images_folder = Path(images_folder)
        self.category_list = categories(self.data)
        self.category_map = {category["id"]: index + 1 for index, category in enumerate(self.category_list)}
        self.by_image = defaultdict(list)
        for annotation in self.data["annotations"]:
            if not annotation.get("iscrowd", 0):
                self.by_image[annotation["image_id"]].append(annotation)

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, index):
        image = self.data["images"][index]
        pixels = pil_to_tensor(load_rgb(image_path(self.images_folder, image["file_name"]))).float() / 255.0
        boxes, labels = [], []
        for annotation in self.by_image[image["id"]]:
            left, top, width, height = annotation["bbox"]
            boxes.append([left, top, left + width, top + height])
            labels.append(self.category_map[annotation["category_id"]])
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": box_tensor, "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image["id"], dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return pixels, target


def collate_batch(batch):
    return tuple(zip(*batch))


def build_model(architecture, num_classes, pretrained=True, image_size=640):
    if architecture == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None, min_size=image_size, max_size=image_size,
            box_score_thresh=0.001, box_detections_per_img=100,
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    elif architecture == "ssd":
        model = ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT if pretrained else None,
            weights_backbone=None, score_thresh=0.001, detections_per_img=100,
        )
        channels = [layer.in_channels for layer in model.head.classification_head.module_list]
        anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(channels, anchors, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def infer_image(model, pixels, device):
    synchronize(device)
    start = time.perf_counter()
    tensor = pil_to_tensor(pixels).float().div(255.0).to(device)
    prediction = model([tensor])[0]
    prediction = {key: value.detach().cpu() for key, value in prediction.items()}
    synchronize(device)
    return prediction, time.perf_counter() - start


def predictions_to_coco(prediction, image_id, category_list):
    records = []
    for box, label, score in zip(prediction["boxes"].tolist(), prediction["labels"].tolist(), prediction["scores"].tolist()):
        if not 1 <= label <= len(category_list):
            raise ValueError(f"Unexpected predicted label: {label}")
        left, top, right, bottom = box
        if right > left and bottom > top:
            records.append({
                "image_id": image_id, "category_id": category_list[label - 1]["id"],
                "bbox": [left, top, right - left, bottom - top], "score": score,
            })
    return records


def predict_dataset(model, data, images_folder, category_list, device, warmup=2):
    model.eval()
    if not data["images"]:
        raise ValueError("Cannot evaluate an empty dataset")
    first = load_rgb(image_path(images_folder, data["images"][0]["file_name"]))
    for iteration in range(warmup):
        infer_image(model, first, device)
    predictions, total_time = [], 0.0
    for image in data["images"]:
        pixels = load_rgb(image_path(images_folder, image["file_name"]))
        output, elapsed = infer_image(model, pixels, device)
        predictions.extend(predictions_to_coco(output, image["id"], category_list))
        total_time += elapsed
    return predictions, total_time / len(data["images"])


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["architecture"], len(checkpoint["categories"]) + 1, pretrained=False, image_size=checkpoint["image_size"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint["categories"], checkpoint["architecture"]


def train(arguments):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    device = torch.device(arguments.device)
    dataset = arguments.dataset
    train_json = arguments.train_annotations or dataset / "annotations/instances_train.json"
    val_json = arguments.val_annotations or dataset / "annotations/instances_val.json"
    training = CocoDetectionDataset(train_json, dataset / "images/train")
    validation = CocoDetectionDataset(val_json, dataset / "images/val")
    if not len(training) or not len(validation):
        raise ValueError("Training and validation splits must be nonempty")
    if [(category["id"], category["name"]) for category in training.category_list] != [(category["id"], category["name"]) for category in validation.category_list]:
        raise ValueError("Training and validation category definitions differ")
    if arguments.output.exists():
        raise ValueError(f"Refusing to overwrite training run: {arguments.output}")
    arguments.output.mkdir(parents=True)
    batch_size = arguments.batch or (4 if arguments.model == "fasterrcnn" else 8)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch, generator=torch.Generator().manual_seed(42))
    model = build_model(arguments.model, len(training.category_list) + 1, pretrained=arguments.pretrained, image_size=arguments.image_size).to(device)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=arguments.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sorted({max(1, int(arguments.epochs * .65)), max(1, int(arguments.epochs * .9))}), gamma=0.1)
    best_map = -1.0
    log_path = arguments.output / "history.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "mAP@0.5", "mAP@0.5:0.95", "precision", "recall"])
        writer.writeheader()
        for epoch in range(1, arguments.epochs + 1):
            model.train()
            running_loss = 0.0
            for images, targets in loader:
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                losses = model(images, targets)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss: {losses}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            predictions, latency = predict_dataset(model, validation.data, validation.images_folder, training.category_list, device, warmup=0)
            metrics = detection_metrics(validation.data, predictions)
            writer.writerow(dict(epoch=epoch, loss=running_loss / len(loader), **metrics))
            stream.flush()
            print(f"Epoch {epoch}/{arguments.epochs}: loss={running_loss / len(loader):.4f}, val mAP={metrics['mAP@0.5:0.95']:.4f}")
            if metrics["mAP@0.5:0.95"] > best_map:
                best_map = metrics["mAP@0.5:0.95"]
                torch.save({
                    "architecture": arguments.model, "model_state": model.state_dict(),
                    "categories": training.category_list, "image_size": arguments.image_size,
                    "epoch": epoch, "validation_metrics": metrics,
                }, arguments.output / "best.pt")
            scheduler.step()
    print(f"Best checkpoint: {arguments.output / 'best.pt'} (validation mAP={best_map:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=["fasterrcnn", "ssd"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--image-size", type=int, default=640, help="Faster R-CNN resize; SSD always uses 300")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    if arguments.epochs <= 0 or (arguments.batch is not None and arguments.batch <= 0):
        parser.error("Epochs and batch size must be positive")
    train(arguments)


if __name__ == "__main__":
    main()