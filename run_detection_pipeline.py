"""Run separate validation, preparation, training and evaluation per modality.

Default stage is validation. Full training is explicit (--stage train or all).
Preparation augments only train to 3x, then exports all splits to YOLO format.
--clip-boxes writes repaired JSON copies, never changes source annotations.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from augment_coco import augment_coco
from coco_to_yolo import convert_dataset, train_yolo
from detection_common import read_coco, validate_coco
from validate_coco import clip_boxes


def validate_dataset(dataset, artifacts):
    artifacts.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split in ("train", "val", "test"):
        report = validate_coco(dataset / "annotations" / f"instances_{split}.json", dataset / "images" / split)
        (artifacts / f"validation_{split}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{dataset.name}/{split}: valid={report['valid']}, images={report['images']}, annotations={report['annotations']}, missing={report['missing_images']}, corrupt={report['corrupt_images']}, invalid_boxes={report['invalid_boxes']}", flush=True)
        reports[split] = report
    return reports


def prepare_dataset(dataset, artifacts, repair=False):
    targets = [artifacts / f"instances_{split}.json" for split in ("train", "val", "test")]
    targets += [artifacts / "instances_train_augmented.json", artifacts / "yolo"]
    if any(target.exists() for target in targets):
        raise ValueError(f"Prepared output already exists in {artifacts}; use stage train/evaluate instead")
    reports = validate_dataset(dataset, artifacts)
    for split, report in reports.items():
        if not report["valid"] and not (repair and report["invalid_boxes"] > 0 and len(report["errors"]) == report["invalid_boxes"]):
            raise ValueError(f"{dataset.name}/{split} failed validation; see {artifacts / f'validation_{split}.json'}")
    for split, report in reports.items():
        source = dataset / "annotations" / f"instances_{split}.json"
        target = artifacts / f"instances_{split}.json"
        if report["valid"]:
            target.write_text(json.dumps(read_coco(source), indent=2) + "\n", encoding="utf-8")
        else:
            changed = clip_boxes(source, target)
            print(f"{dataset.name}/{split}: clipped {changed} boxes into a new JSON copy", flush=True)
    augmented_json = artifacts / "instances_train_augmented.json"
    data = augment_coco(artifacts / "instances_train.json", dataset / "images/train", augmented_json, factor=3, seed=42)
    if not validate_coco(augmented_json, dataset / "images/train")["valid"]:
        raise ValueError("Augmentation output failed integrity validation")
    print(f"{dataset.name}: augmented training count = {len(data['images'])}", flush=True)
    config = convert_dataset(dataset, artifacts / "yolo", augmented_json, artifacts / "instances_val.json", artifacts / "instances_test.json")
    print(f"{dataset.name}: YOLO export saved to {config}", flush=True)


def run_command(script, arguments):
    subprocess.run([sys.executable, str(Path(__file__).parent / script), *map(str, arguments)], check=True)


def train_dataset(dataset, artifacts, device):
    config = artifacts / "yolo/data.yaml"
    if not config.is_file():
        raise ValueError(f"Run prepare first: missing {config}")
    models = artifacts / "models"
    train_yolo(config, models, epochs=50, imgsz=640, batch=8, device=device)
    for architecture, batch in (("fasterrcnn", 4), ("ssd", 8)):
        run_command("train_torchvision.py", [
            "--dataset", dataset, "--model", architecture, "--output", models / architecture,
            "--train-annotations", artifacts / "instances_train_augmented.json",
            "--val-annotations", artifacts / "instances_val.json",
            "--epochs", 20, "--batch", batch, "--device", device,
        ])


def evaluate_dataset(dataset, artifacts, device):
    models = artifacts / "models"
    run_command("evaluate_models.py", [
        "--annotations", artifacts / "instances_test.json", "--images", dataset / "images/test",
        "--yolo", models / "yolov8s/weights/best.pt",
        "--yolo-mapping", artifacts / "yolo/category_mapping.json",
        "--fasterrcnn", models / "fasterrcnn/best.pt", "--ssd", models / "ssd/best.pt",
        "--output", artifacts / "results", "--device", device,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=[Path("RGB_Dataset"), Path("Thermal_Dataset")])
    parser.add_argument("--output", type=Path, default=Path("detection_runs"))
    parser.add_argument("--stage", choices=["validate", "prepare", "train", "evaluate", "all"], default="validate")
    parser.add_argument("--clip-boxes", action="store_true", help="Explicitly permit clipping into new JSON copies during preparation")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; the same device is used for all models")
    arguments = parser.parse_args()
    if len({dataset.resolve().name for dataset in arguments.datasets}) != len(arguments.datasets):
        parser.error("Dataset directory names must be unique")
    all_valid = True
    for dataset in arguments.datasets:
        dataset = dataset.resolve()
        artifacts = arguments.output.resolve() / dataset.name
        if arguments.stage == "validate":
            reports = validate_dataset(dataset, artifacts)
            all_valid = all_valid and all(report["valid"] for report in reports.values())
        if arguments.stage in ("prepare", "all"):
            prepare_dataset(dataset, artifacts, arguments.clip_boxes)
        if arguments.stage in ("train", "all"):
            train_dataset(dataset, artifacts, arguments.device)
        if arguments.stage in ("evaluate", "all"):
            evaluate_dataset(dataset, artifacts, arguments.device)
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()