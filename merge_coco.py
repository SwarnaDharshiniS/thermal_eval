"""Merge COCO datasets and copy their images using only the standard library.

Python API:
    merge_coco([(coco_json_path, images_folder_path), ...], output_folder)

CLI (repeat --dataset for each input):
    python merge_coco.py --dataset source1/annotations.json source1/images \
        --dataset source2/annotations.json source2/images --output merged

Each file_name must be relative to its supplied images folder. Categories are
merged by exact name (not numeric ID); the first category's metadata is retained.
Different spellings remain separate classes. Core COCO records and licenses are
preserved, but dataset-level info and custom top-level fields are not merged.
Output contains annotations.json and images/ with uniquely numbered filenames.
Existing output files are never overwritten.
"""

import argparse
import json
from pathlib import Path
import shutil


def merge_coco(
    datasets: list[tuple[str | Path, str | Path]], output_folder: str | Path
) -> dict:
    if not datasets:
        raise ValueError("Provide at least one (COCO JSON, images folder) pair")
    output_folder = Path(output_folder).resolve()
    output_json = output_folder / "annotations.json"
    if output_json.exists():
        raise ValueError(f"Refusing to overwrite {output_json}")

    merged = {"categories": [], "images": [], "annotations": []}
    categories_by_name = {}
    merged_licenses = []
    copy_plan = []

    for json_path, images_folder in datasets:
        json_path = Path(json_path)
        images_folder = Path(images_folder).resolve()
        if not images_folder.is_dir():
            raise ValueError(f"Image directory does not exist: {images_folder}")
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        for field in ("categories", "images", "annotations"):
            if not isinstance(data.get(field), list):
                raise ValueError(f"{json_path}: {field} must be a list")

        category_map = {}
        for category in data["categories"]:
            old_id = category["id"]
            if old_id in category_map:
                raise ValueError(f"{json_path}: duplicate category ID {old_id}")
            name = category["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{json_path}: category names must be nonempty strings")
            if name not in categories_by_name:
                new_category = dict(category, id=len(merged["categories"]) + 1)
                merged["categories"].append(new_category)
                categories_by_name[name] = new_category
            existing = categories_by_name[name]
            for field in ("keypoints", "skeleton"):
                if existing.get(field) != category.get(field):
                    raise ValueError(f"{json_path}: incompatible {field} for category {name}")
            category_map[old_id] = existing["id"]

        license_map = {}
        for license_record in data.get("licenses", []):
            old_id = license_record["id"]
            if old_id in license_map:
                raise ValueError(f"{json_path}: duplicate license ID {old_id}")
            new_id = len(merged_licenses) + 1
            merged_licenses.append(dict(license_record, id=new_id))
            license_map[old_id] = new_id

        image_map = {}
        for image in data["images"]:
            old_id = image["id"]
            if old_id in image_map:
                raise ValueError(f"{json_path}: duplicate image ID {old_id}")
            relative_path = Path(image["file_name"])
            source = (images_folder / relative_path).resolve()
            if relative_path.is_absolute() or not source.is_relative_to(images_folder):
                raise ValueError(f"{json_path}: image path must stay within {images_folder}: {relative_path}")
            if not source.is_file():
                raise FileNotFoundError(f"{json_path}: image not found: {source}")
            new_id = len(merged["images"]) + 1
            destination_name = f"{new_id:08d}{source.suffix.lower()}"
            destination = output_folder / "images" / destination_name
            if destination.exists():
                raise ValueError(f"Refusing to overwrite {destination}")
            new_image = dict(image, id=new_id, file_name=f"images/{destination_name}")
            if "license" in new_image:
                if new_image["license"] not in license_map:
                    raise ValueError(f"{json_path}: image {old_id} references an undefined license")
                new_image["license"] = license_map[new_image["license"]]
            merged["images"].append(new_image)
            image_map[old_id] = new_id
            copy_plan.append((source, destination))

        annotation_ids = set()
        for annotation in data["annotations"]:
            old_id = annotation["id"]
            if old_id in annotation_ids:
                raise ValueError(f"{json_path}: duplicate annotation ID {old_id}")
            annotation_ids.add(old_id)
            if annotation["image_id"] not in image_map:
                raise ValueError(f"{json_path}: annotation {old_id} references an undefined image")
            if annotation["category_id"] not in category_map:
                raise ValueError(f"{json_path}: annotation {old_id} references an undefined category")
            merged["annotations"].append(dict(
                annotation,
                id=len(merged["annotations"]) + 1,
                image_id=image_map[annotation["image_id"]],
                category_id=category_map[annotation["category_id"]],
            ))

    if merged_licenses:
        merged["licenses"] = merged_licenses
    serialized = json.dumps(merged, indent=2, allow_nan=False) + "\n"
    (output_folder / "images").mkdir(parents=True, exist_ok=True)
    for source, destination in copy_plan:
        with source.open("rb") as source_file, destination.open("xb") as destination_file:
            shutil.copyfileobj(source_file, destination_file)
    with output_json.open("x", encoding="utf-8") as output_file:
        output_file.write(serialized)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", nargs=2, action="append", required=True,
        metavar=("COCO_JSON", "IMAGES_FOLDER"), help="Input pair; repeat for each dataset",
    )
    parser.add_argument("--output", type=Path, required=True, help="Merged output folder")
    arguments = parser.parse_args()
    try:
        merged = merge_coco(
            [(Path(json_path), Path(images_folder)) for json_path, images_folder in arguments.dataset],
            arguments.output,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(
        f"Wrote {arguments.output / 'annotations.json'}: {len(merged['images'])} images, "
        f"{len(merged['annotations'])} annotations, {len(merged['categories'])} categories"
    )


if __name__ == "__main__":
    main()