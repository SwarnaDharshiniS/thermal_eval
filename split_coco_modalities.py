"""Separate the workspace's merged COCO data into seeded RGB/thermal splits.

Run: python split_coco_modalities.py --root . --seed 42

Source order must match the merge order. Override defaults with repeated
--source RGB|Thermal COCO_JSON IMAGES_FOLDER arguments, in merge order.
Paths supplied via --source and --merged are relative to --root.
Image file_name values in each split JSON are relative to images/<split>/.
Training and validation counts are floored; test receives the remainder.
Source files remain untouched. Existing destination folders are rejected.
"""

import argparse
from collections import defaultdict
import filecmp
import json
from pathlib import Path
import random
import shutil


def default_sources(root: Path) -> list[tuple[str, Path, Path]]:
    sources = [("Thermal", root / "annotations.json", root)]
    for folder, modality in (("images-rgb.coco", "RGB"), ("thermal.coco", "Thermal")):
        for split in ("test", "train", "valid"):
            images_folder = root / folder / split
            sources.append((modality, images_folder / "_annotations.coco.json", images_folder))
    return sources


def separate_and_split(
    coco_json: Path,
    sources: list[tuple[str, Path, Path]],
    output_root: Path,
    seed: int = 42,
) -> dict:
    destinations = {name: output_root / f"{name}_Dataset" for name in ("RGB", "Thermal")}
    for destination in destinations.values():
        if destination.exists():
            raise ValueError(f"Refusing to overwrite existing folder: {destination}")

    data = json.loads(coco_json.read_text(encoding="utf-8-sig"))
    grouped_images = {name: [] for name in destinations}
    category_names = {name: set() for name in destinations}
    image_index = 0
    for modality, source_json, source_folder in sources:
        if modality not in destinations:
            raise ValueError(f"Unknown modality: {modality}")
        original = json.loads(source_json.read_text(encoding="utf-8-sig"))
        category_names[modality].update(category["name"] for category in original["categories"])
        for original_image in original["images"]:
            if image_index >= len(data["images"]):
                raise ValueError("Sources contain more images than the merged JSON")
            merged_image = data["images"][image_index]
            expected_id = image_index + 1
            source_path = source_folder / original_image["file_name"]
            merged_path = coco_json.parent / merged_image["file_name"]
            if merged_image["id"] != expected_id or any(
                merged_image[field] != original_image[field] for field in ("width", "height")
            ):
                raise ValueError(f"Source order does not match merged image {expected_id}")
            if not filecmp.cmp(source_path, merged_path, shallow=False):
                raise ValueError(f"Merged image {expected_id} does not match source {source_path}")
            grouped_images[modality].append(merged_image)
            image_index += 1
    if image_index != len(data["images"]):
        raise ValueError("Sources do not account for all merged images")
    if any(not images for images in grouped_images.values()):
        raise ValueError("Both RGB and Thermal sources must contain images")

    image_ids = {image["id"] for image in data["images"]}
    category_ids = {category["id"] for category in data["categories"]}
    annotations_by_image = defaultdict(list)
    annotation_ids = set()
    for annotation in data["annotations"]:
        if annotation["id"] in annotation_ids:
            raise ValueError(f"Duplicate annotation ID: {annotation['id']}")
        if annotation["image_id"] not in image_ids or annotation["category_id"] not in category_ids:
            raise ValueError(f"Invalid reference in annotation {annotation['id']}")
        annotation_ids.add(annotation["id"])
        annotations_by_image[annotation["image_id"]].append(annotation)

    plans = []
    counts = {}
    for modality, images in grouped_images.items():
        random.Random(seed).shuffle(images)
        train_end = int(len(images) * 0.70)
        val_end = train_end + int(len(images) * 0.15)
        partitions = {"train": images[:train_end], "val": images[train_end:val_end], "test": images[val_end:]}
        categories = [category for category in data["categories"] if category["name"] in category_names[modality]]
        allowed_categories = {category["id"] for category in categories}
        counts[modality] = {}
        for split, split_images in partitions.items():
            destination = destinations[modality]
            split_annotations = [
                annotation for image in split_images
                for annotation in annotations_by_image[image["id"]]
            ]
            if any(annotation["category_id"] not in allowed_categories for annotation in split_annotations):
                raise ValueError(f"Category provenance does not match {modality} sources")
            split_data = dict(data)
            split_data["categories"] = categories
            split_data["images"] = [
                dict(image, file_name=Path(image["file_name"]).name) for image in split_images
            ]
            split_data["annotations"] = split_annotations
            serialized = json.dumps(split_data, indent=2, allow_nan=False) + "\n"
            copies = [
                (coco_json.parent / image["file_name"], destination / "images" / split / Path(image["file_name"]).name)
                for image in split_images
            ]
            if len({target for source, target in copies}) != len(copies):
                raise ValueError(f"Duplicate filenames in {modality}/{split}")
            plans.append((destination, split, serialized, copies))
            counts[modality][split] = (len(split_images), len(split_annotations))

    for destination in destinations.values():
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "annotations").mkdir()
    for destination, split, serialized, copies in plans:
        (destination / "images" / split).mkdir(parents=True)
        for source, target in copies:
            shutil.copyfile(source, target)
        (destination / "annotations" / f"instances_{split}.json").write_text(serialized, encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--merged", type=Path, default=Path("merged/annotations.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", nargs=3, action="append", metavar=("MODALITY", "COCO_JSON", "IMAGES_FOLDER"))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    sources = (
        [(name, root / json_path, root / images_folder) for name, json_path, images_folder in arguments.source]
        if arguments.source else default_sources(root)
    )
    try:
        counts = separate_and_split(root / arguments.merged, sources, root, arguments.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    for modality, splits in counts.items():
        for split, (images, annotations) in splits.items():
            print(f"{modality}_Dataset/{split}: {images} images, {annotations} annotations")


if __name__ == "__main__":
    main()