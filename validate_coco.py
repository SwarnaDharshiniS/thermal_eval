"""Validate a COCO JSON and all images in its image folder."""

import argparse
import json
import math
from pathlib import Path

from detection_common import read_coco, validate_coco


def clip_boxes(json_path, output):
    if Path(output).exists():
        raise ValueError(f"Refusing to overwrite {output}")
    data = read_coco(json_path)
    images = {image["id"]: image for image in data["images"]}
    changed = 0
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        left, top, width, height = annotation["bbox"]
        if not all(math.isfinite(value) for value in (left, top, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"Cannot clip invalid box for annotation {annotation['id']}")
        if left >= 0 and top >= 0 and left + width <= image["width"] and top + height <= image["height"]:
            continue
        right = min(image["width"], left + width)
        bottom = min(image["height"], top + height)
        left, top = max(0, left), max(0, top)
        clipped = [left, top, right - left, bottom - top]
        if clipped[2] <= 0 or clipped[3] <= 0:
            raise ValueError(f"Annotation {annotation['id']} falls entirely outside its image")
        if clipped != annotation["bbox"]:
            if annotation.get("segmentation") or annotation.get("keypoints"):
                raise ValueError("Box clipping does not repair segmentation/keypoint annotations")
            annotation["bbox"] = clipped
            annotation["area"] = clipped[2] * clipped[3]
            changed += 1
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--clip-boxes-output", type=Path, help="Optional new JSON with boxes clipped to image bounds; never overwrites originals")
    arguments = parser.parse_args()
    report = validate_coco(arguments.annotations, arguments.images)
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))
    for error in report["errors"][:20]:
        print(f"ERROR: {error}")
    if len(report["errors"]) > 20:
        print(f"... {len(report['errors']) - 20} more errors (use --report for the full report)")
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if arguments.clip_boxes_output:
        changed = clip_boxes(arguments.annotations, arguments.clip_boxes_output)
        fixed = validate_coco(arguments.clip_boxes_output, arguments.images)
        print(f"Clipped {changed} boxes into {arguments.clip_boxes_output}; valid={fixed['valid']}")
        raise SystemExit(0 if fixed["valid"] else 1)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()