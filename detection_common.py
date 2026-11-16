"""Shared COCO validation, category mapping, and detection metrics."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def read_coco(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for field in ("images", "annotations", "categories"):
        if not isinstance(data.get(field), list):
            raise ValueError(f"{path}: {field} must be a list")
    return data


def image_path(images_folder, file_name):
    root = Path(images_folder).resolve()
    relative = Path(file_name)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Image path escapes the images folder: {file_name}")
    return path


def categories(data):
    return sorted(data["categories"], key=lambda category: category["id"])


def validate_coco(json_path, images_folder):
    data = read_coco(json_path)
    errors = []
    duplicate_images = [key for key, count in Counter(image["id"] for image in data["images"]).items() if count > 1]
    duplicate_annotations = [key for key, count in Counter(item["id"] for item in data["annotations"]).items() if count > 1]
    duplicate_categories = [key for key, count in Counter(item["id"] for item in data["categories"]).items() if count > 1]
    for name, duplicates in (("image", duplicate_images), ("annotation", duplicate_annotations), ("category", duplicate_categories)):
        if duplicates:
            errors.append(f"Duplicate {name} IDs: {duplicates}")
    images = {image["id"]: image for image in data["images"]}
    category_ids = {category["id"] for category in data["categories"]}
    missing = corrupt = dimension_mismatches = invalid_boxes = invalid_references = 0
    referenced = set()
    for image in data["images"]:
        try:
            path = image_path(images_folder, image["file_name"])
            referenced.add(path)
            if not path.is_file():
                missing += 1
                errors.append(f"Missing image {image['id']}: {path}")
                continue
            with Image.open(path) as decoded:
                decoded.load()
                oriented = ImageOps.exif_transpose(decoded)
                if oriented.size != (image["width"], image["height"]):
                    dimension_mismatches += 1
                    errors.append(f"Image {image['id']}: JSON dimensions differ from oriented size {oriented.size}")
        except (OSError, ValueError) as error:
            corrupt += 1
            errors.append(f"Unreadable image {image['id']}: {error}")
    for path in Path(images_folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"} and path.resolve() not in referenced:
            try:
                with Image.open(path) as decoded:
                    decoded.load()
            except (OSError, ValueError) as error:
                corrupt += 1
                errors.append(f"Unreadable unreferenced image {path}: {error}")
    for annotation in data["annotations"]:
        image = images.get(annotation.get("image_id"))
        if image is None or annotation.get("category_id") not in category_ids:
            invalid_references += 1
            errors.append(f"Annotation {annotation['id']}: undefined image or category")
            continue
        box = annotation.get("bbox", [])
        valid = isinstance(box, list) and len(box) == 4 and all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in box
        )
        if valid:
            left, top, width, height = box
            valid = (
                left >= -1e-6 and top >= -1e-6 and width > 0 and height > 0
                and left + width <= image["width"] + 1e-6
                and top + height <= image["height"] + 1e-6
            )
        if not valid:
            invalid_boxes += 1
            errors.append(f"Annotation {annotation['id']}: invalid/out-of-bounds bbox {box}")
    return {
        "json": str(json_path), "images": len(data["images"]),
        "annotations": len(data["annotations"]), "categories": len(category_ids),
        "missing_images": missing, "corrupt_images": corrupt,
        "dimension_mismatches": dimension_mismatches, "invalid_boxes": invalid_boxes,
        "invalid_references": invalid_references,
        "duplicate_image_ids": duplicate_images, "duplicate_annotation_ids": duplicate_annotations,
        "duplicate_category_ids": duplicate_categories, "valid": not errors, "errors": errors,
    }


def require_valid(json_path, images_folder):
    report = validate_coco(json_path, images_folder)
    if not report["valid"]:
        raise ValueError(f"Invalid dataset {json_path}: " + "; ".join(report["errors"][:5]))
    return read_coco(json_path)


def bbox_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def detection_metrics(data, predictions, confidence=0.25):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO()
    ground_truth.dataset = dict(data, info=data.get("info", {}))
    ground_truth.dataset["annotations"] = [
        dict(item, area=item.get("area", item["bbox"][2] * item["bbox"][3]), iscrowd=item.get("iscrowd", 0))
        for item in data["annotations"]
    ]
    ground_truth.createIndex()
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {"images": data["images"], "categories": data["categories"], "annotations": []}
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = [image["id"] for image in data["images"]]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    selected = sorted((item for item in predictions if item["score"] >= confidence), key=lambda item: item["score"], reverse=True)
    truths = defaultdict(list)
    for annotation in data["annotations"]:
        truths[(annotation["image_id"], annotation["category_id"])].append(annotation)
    matched = set()
    true_positive = false_positive = 0
    for prediction in selected:
        candidates = truths[(prediction["image_id"], prediction["category_id"])]
        regular = [item for item in candidates if not item.get("iscrowd", 0) and item["id"] not in matched]
        best = max(regular, key=lambda item: bbox_iou(prediction["bbox"], item["bbox"]), default=None)
        if best is not None and bbox_iou(prediction["bbox"], best["bbox"]) >= 0.5:
            matched.add(best["id"])
            true_positive += 1
        elif not any(
            item.get("iscrowd", 0) and crowd_overlap(prediction["bbox"], item["bbox"]) >= 0.5 for item in candidates
        ):
            false_positive += 1
    total = sum(not annotation.get("iscrowd", 0) for annotation in data["annotations"])
    return {
        "mAP@0.5": max(0.0, float(evaluator.stats[1])),
        "mAP@0.5:0.95": max(0.0, float(evaluator.stats[0])),
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, total),
    }


def crowd_overlap(box, crowd):
    intersection = max(0, min(box[0] + box[2], crowd[0] + crowd[2]) - max(box[0], crowd[0])) * max(
        0, min(box[1] + box[3], crowd[1] + crowd[3]) - max(box[1], crowd[1])
    )
    return intersection / max(1e-12, box[2] * box[3])