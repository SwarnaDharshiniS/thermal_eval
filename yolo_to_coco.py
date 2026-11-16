"""Convert YOLO detection labels to COCO; requires Pillow (pip install Pillow).

Example:
    python yolo_to_coco.py --images dataset/images --labels dataset/labels \
        --classes dataset/classes.txt --output dataset/annotations.json

Label paths mirror image paths, with .txt replacing the image extension.
Missing or empty labels represent images without objects. Segmentation and
oriented bounding-box labels are not supported. Image file names are relative
to the output JSON's directory, including ../ when necessary.
"""

import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def convert(images_dir: Path, labels_dir: Path, classes_file: Path, output: Path) -> dict:
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()
    output = output.resolve()
    if not images_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"Label directory does not exist: {labels_dir}")

    class_names = classes_file.read_text(encoding="utf-8-sig").splitlines()
    class_names = [name.strip() for name in class_names]
    if not class_names or any(not name for name in class_names):
        raise ValueError("Classes file must contain one nonempty class name per line")

    coco = {
        "categories": [
            {"id": class_id + 1, "name": name, "supercategory": "none"}
            for class_id, name in enumerate(class_names)
        ],
        "images": [],
        "annotations": [],
    }
    image_paths = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {images_dir}")

    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        file_name = Path(os.path.relpath(image_path, output.parent)).as_posix()
        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": image_width,
            "height": image_height,
        })
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        if not label_path.exists():
            continue

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            location = f"{label_path}:{line_number}"
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{location}: expected 5 fields for a detection box")
            try:
                class_id = int(fields[0])
                center_x, center_y, width, height = map(float, fields[1:])
            except ValueError as error:
                raise ValueError(f"{location}: invalid class ID or box coordinates") from error
            if not 0 <= class_id < len(class_names):
                raise ValueError(f"{location}: class ID {class_id} is not in the classes file")
            coordinates = (center_x, center_y, width, height)
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
                raise ValueError(f"{location}: coordinates must be finite and normalized to [0, 1]")
            if width == 0 or height == 0:
                raise ValueError(f"{location}: box width and height must be positive")

            box_width = width * image_width
            box_height = height * image_height
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [
                    (center_x - width / 2) * image_width,
                    (center_y - height / 2) * image_height,
                    box_width,
                    box_height,
                ],
                "area": box_width * box_height,
                "iscrowd": 0,
                "segmentation": [],
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Image directory (searched recursively)")
    parser.add_argument("--labels", type=Path, required=True, help="Matching YOLO label directory")
    parser.add_argument("--classes", type=Path, required=True, help="Class names in YOLO class-ID order")
    parser.add_argument("--output", type=Path, default=Path("annotations.json"), help="Output COCO JSON")
    arguments = parser.parse_args()
    try:
        coco = convert(arguments.images, arguments.labels, arguments.classes, arguments.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(
        f"Wrote {arguments.output}: {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, {len(coco['categories'])} categories"
    )


if __name__ == "__main__":
    main()