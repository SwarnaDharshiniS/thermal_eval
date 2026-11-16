"""Focused tests using synthetic images; no pretrained-weight downloads."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from augment_coco import augment_coco
from coco_to_yolo import convert_dataset
from detection_common import detection_metrics, validate_coco
from validate_coco import clip_boxes


class DetectionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.data = {
            "categories": [{"id": 2, "name": "normal"}, {"id": 9, "name": "fault"}],
            "images": [
                {"id": 11, "file_name": "object.png", "width": 80, "height": 64},
                {"id": 12, "file_name": "background.png", "width": 80, "height": 64},
            ],
            "annotations": [{"id": 21, "image_id": 11, "category_id": 9, "bbox": [16, 16, 24, 16], "area": 384, "iscrowd": 0, "segmentation": []}],
        }
        (self.dataset / "annotations").mkdir(parents=True)
        for split in ("train", "val", "test"):
            folder = self.dataset / "images" / split
            folder.mkdir(parents=True)
            Image.new("RGB", (80, 64), (20, 140, 70)).save(folder / "object.png")
            Image.new("RGB", (80, 64), (0, 0, 0)).save(folder / "background.png")
            self.write_data(split, self.data)

    def tearDown(self):
        self.temporary.cleanup()

    def write_data(self, split, data):
        path = self.dataset / "annotations" / f"instances_{split}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_validation_and_non_destructive_clipping(self):
        path = self.dataset / "annotations/instances_train.json"
        folder = self.dataset / "images/train"
        self.assertTrue(validate_coco(path, folder)["valid"])
        broken = json.loads(json.dumps(self.data))
        broken["annotations"][0]["bbox"] = [70, 16, 24, 16]
        self.write_data("train", broken)
        self.assertEqual(validate_coco(path, folder)["invalid_boxes"], 1)
        output = self.root / "clipped.json"
        self.assertEqual(clip_boxes(path, output), 1)
        self.assertTrue(validate_coco(output, folder)["valid"])
        self.assertEqual(json.loads(path.read_text())["annotations"][0]["bbox"][2], 24)
        broken["images"].append(dict(broken["images"][0]))
        broken["annotations"].append(dict(broken["annotations"][0]))
        self.write_data("train", broken)
        (folder / "object.png").unlink()
        (folder / "background.png").write_bytes(b"corrupt image")
        report = validate_coco(path, folder)
        self.assertEqual(report["duplicate_image_ids"], [11])
        self.assertEqual(report["duplicate_annotation_ids"], [21])
        self.assertGreater(report["missing_images"], 0)
        self.assertGreater(report["corrupt_images"], 0)

    def test_exif_orientation(self):
        folder = self.dataset / "images/train"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (64, 80)).save(folder / "object.jpg", exif=exif)
        data = json.loads(json.dumps(self.data))
        data["images"][0]["file_name"] = "object.jpg"
        path = self.write_data("train", data)
        self.assertTrue(validate_coco(path, folder)["valid"])

    def test_threefold_augmentation(self):
        path = self.dataset / "annotations/instances_train.json"
        folder = self.dataset / "images/train"
        output = self.root / "augmented.json"
        merged = augment_coco(path, folder, output, factor=3, seed=42)
        self.assertEqual(len(merged["images"]), 6)
        self.assertEqual(len({image["id"] for image in merged["images"]}), 6)
        self.assertEqual(len({item["id"] for item in merged["annotations"]}), len(merged["annotations"]))
        self.assertTrue(validate_coco(output, folder)["valid"])
        self.assertEqual(len(json.loads(path.read_text())["images"]), 2)
        with self.assertRaises(ValueError):
            augment_coco(path, folder, self.root / "another.json")

    def test_yolo_conversion(self):
        import yaml
        output = self.root / "yolo"
        config = convert_dataset(self.dataset, output)
        settings = yaml.safe_load(config.read_text())
        self.assertEqual(settings["names"], ["normal", "fault"])
        fields = (output / "labels/train/00000011.txt").read_text().split()
        self.assertEqual(fields[0], "1")
        self.assertEqual([float(value) for value in fields[1:]], [.35, .375, .3, .25])
        self.assertEqual((output / "labels/train/00000012.txt").read_text(), "")
        with Image.open(output / "images/train/00000011.png") as image:
            self.assertEqual(image.size, (80, 64))

    def test_yolo_training_and_common_inference(self):
        import torch
        from ultralytics import YOLO
        from evaluate_models import predict_yolo
        torch.set_num_threads(2)
        config = convert_dataset(self.dataset, self.root / "yolo")
        model = YOLO("yolov8s.yaml")
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(
                data=str(config), project=str(self.root / "runs"), name="smoke",
                epochs=1, imgsz=64, batch=2, workers=0, device="cpu", pretrained=False,
                plots=False, amp=False, verbose=False, seed=42,
            )
        checkpoint = Path(model.trainer.best)
        self.assertTrue(checkpoint.is_file())
        predictions, latency = predict_yolo(
            checkpoint, self.root / "yolo/category_mapping.json", self.data,
            self.dataset / "images/test", "cpu", imgsz=64,
        )
        self.assertGreater(latency, 0)
        self.assertTrue(all(item["category_id"] in {2, 9} for item in predictions))

    def test_common_metrics_and_comparison_outputs(self):
        from evaluate_models import save_comparison
        prediction = {"image_id": 11, "category_id": 9, "bbox": [16, 16, 24, 16], "score": .99}
        with contextlib.redirect_stdout(io.StringIO()):
            perfect = detection_metrics(self.data, [prediction])
            empty = detection_metrics(self.data, [])
            duplicate = detection_metrics(self.data, [prediction, dict(prediction, score=.8)])
        self.assertAlmostEqual(perfect["mAP@0.5:0.95"], 1.0)
        self.assertEqual(perfect["precision"], 1.0)
        self.assertEqual(perfect["recall"], 1.0)
        self.assertEqual(empty["mAP@0.5"], 0.0)
        self.assertEqual(duplicate["precision"], .5)
        output = self.root / "results"
        save_comparison([dict(model=name, **perfect, inference_ms_per_image=10) for name in ["YOLOv8s", "Faster R-CNN", "SSD300"]], output)
        self.assertTrue((output / "comparison.csv").is_file())
        with Image.open(output / "results.png") as image:
            self.assertGreater(image.width, 0)

    def test_torchvision_training_and_checkpoint_reload(self):
        import torch
        from train_torchvision import load_model, train
        torch.set_num_threads(2)
        for architecture in ("fasterrcnn", "ssd"):
            output = self.root / architecture
            arguments = argparse.Namespace(
                dataset=self.dataset, model=architecture, output=output,
                train_annotations=None, val_annotations=None, epochs=1, batch=2,
                image_size=64, device="cpu", lr=.005, pretrained=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                train(arguments)
            self.assertTrue((output / "best.pt").is_file())
            model, category_list, loaded_architecture = load_model(output / "best.pt", "cpu")
            self.assertEqual(loaded_architecture, architecture)
            self.assertEqual([category["id"] for category in category_list], [2, 9])
            self.assertFalse(model.training)
            del model


if __name__ == "__main__":
    unittest.main()