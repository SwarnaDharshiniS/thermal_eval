"""Convert train/val/test COCO splits to YOLOv8 and optionally train YOLOv8s."""

import argparse
import json
from pathlib import Path

from detection_common import categories, image_path, load_rgb, require_valid


def convert_dataset(dataset, output, train_annotations=None, val_annotations=None, test_annotations=None):
    import yaml

    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    overrides = {"train": train_annotations, "val": val_annotations, "test": test_annotations}
    splits = {
        split: require_valid(overrides[split] or dataset / "annotations" / f"instances_{split}.json", dataset / "images" / split)
        for split in overrides
    }
    category_list = categories(splits["train"])
    category_signature = [(item["id"], item["name"]) for item in category_list]
    if any([(item["id"], item["name"]) for item in categories(data)] != category_signature for data in splits.values()):
        raise ValueError("Train, validation, and test category definitions must match")
    category_map = {item["id"]: index for index, item in enumerate(category_list)}
    for split, data in splits.items():
        annotations_by_image = {}
        for annotation in data["annotations"]:
            if not annotation.get("iscrowd", 0):
                annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        for image in data["images"]:
            stem = f"{image['id']:08d}"
            load_rgb(image_path(dataset / "images" / split, image["file_name"])).save(output / "images" / split / f"{stem}.png")
            labels = []
            for annotation in annotations_by_image.get(image["id"], []):
                left, top, width, height = annotation["bbox"]
                values = [
                    (left + width / 2) / image["width"], (top + height / 2) / image["height"],
                    width / image["width"], height / image["height"],
                ]
                labels.append(f"{category_map[annotation['category_id']]} " + " ".join(f"{value:.10f}" for value in values))
            (output / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    mapping = {"categories": category_list}
    (output / "category_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": [item["name"] for item in category_list]}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output / "data.yaml"


def train_yolo(config, project, name="yolov8s", epochs=50, imgsz=640, batch=8, device="cpu"):
    from ultralytics import YOLO
    if (Path(project) / name).exists():
        raise ValueError(f"Refusing to overwrite training run: {Path(project) / name}")
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(config), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, project=str(project), name=name,
        seed=42, deterministic=True, workers=0, save=True, patience=epochs,
    )
    print(f"Best checkpoint: {model.trainer.best}")
    return model.trainer.best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-annotations", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/detection"))
    parser.add_argument("--name", default="yolov8s")
    arguments = parser.parse_args()
    config = convert_dataset(arguments.dataset, arguments.output, arguments.train_annotations, arguments.val_annotations, arguments.test_annotations)
    print(f"Created {config}")
    if arguments.train:
        train_yolo(config, arguments.project, arguments.name, arguments.epochs, arguments.imgsz, arguments.batch, arguments.device)


if __name__ == "__main__":
    main()