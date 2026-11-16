"""Offline Albumentations detection augmentation; total images = factor * original.

Only use this on a training split. Original JSON is preserved; output must be
a new JSON path. Generated images are saved beside the original images.
Existing generated filenames are never overwritten. Segmentation/keypoint
annotations are rejected because this workflow transforms bounding boxes only.
"""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

from detection_common import image_path, load_rgb, require_valid


def augment_coco(annotations, images_folder, output, factor=3, seed=42):
    import albumentations as augmentation
    import cv2
    import numpy as np

    if factor < 1 or not isinstance(factor, int):
        raise ValueError("Factor must be a positive integer (3 means original plus two variants)")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    original = require_valid(annotations, images_folder)
    if any(item.get("segmentation") or item.get("keypoints") for item in original["annotations"]):
        raise ValueError("This script supports detection boxes only, not segmentation or keypoints")
    if not original["images"]:
        raise ValueError("Cannot augment an empty dataset")
    by_image = defaultdict(list)
    for annotation in original["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    for image in original["images"]:
        for variant in range(1, factor):
            target = Path(images_folder) / f"aug_{image['id']}_{variant}.png"
            if target.exists():
                raise ValueError(f"Refusing to overwrite {target}")
    transform = augmentation.Compose([
        augmentation.HorizontalFlip(p=0.5),
        augmentation.RandomBrightnessContrast(p=0.7),
        augmentation.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        augmentation.GaussianBlur(blur_limit=(3, 5), p=0.3),
        augmentation.HueSaturationValue(p=0.5),
    ], bbox_params=augmentation.BboxParams(
        format="coco", label_fields=["annotation_indices"], clip=True,
        filter_invalid_bboxes=True, min_visibility=0.1,
    ), seed=seed)
    merged = copy.deepcopy(original)
    next_image_id = max(image["id"] for image in original["images"]) + 1
    next_annotation_id = max((item["id"] for item in original["annotations"]), default=0) + 1
    for image in original["images"]:
        pixels = np.asarray(load_rgb(image_path(images_folder, image["file_name"])))
        annotations_for_image = by_image[image["id"]]
        for variant in range(1, factor):
            result = transform(
                image=pixels, bboxes=[item["bbox"] for item in annotations_for_image],
                annotation_indices=list(range(len(annotations_for_image))),
            )
            file_name = f"aug_{image['id']}_{variant}.png"
            from PIL import Image
            Image.fromarray(result["image"]).save(Path(images_folder) / file_name)
            merged["images"].append(dict(image, id=next_image_id, file_name=file_name))
            for box, index in zip(result["bboxes"], result["annotation_indices"]):
                old_annotation = annotations_for_image[int(index)]
                left, top, width, height = map(float, box)
                left = max(0.0, min(left, image["width"]))
                top = max(0.0, min(top, image["height"]))
                width = min(width, image["width"] - left)
                height = min(height, image["height"] - top)
                if width <= 0 or height <= 0:
                    continue
                merged["annotations"].append(dict(
                    old_annotation, id=next_annotation_id, image_id=next_image_id,
                    bbox=[left, top, width, height], area=width * height, segmentation=[],
                ))
                next_annotation_id += 1
            next_image_id += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    data = augment_coco(arguments.annotations, arguments.images, arguments.output, arguments.factor, arguments.seed)
    print(f"Wrote {arguments.output}: {len(data['images'])} images, {len(data['annotations'])} annotations")


if __name__ == "__main__":
    main()