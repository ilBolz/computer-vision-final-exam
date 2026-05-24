"""
Build classifier dataset from VisDrone by extracting vehicle/pedestrian crops.

Uses ground-truth bounding boxes to crop and resize regions to 224x224.
Output structure:
    data/processed/classifier/{train,val,test}/{class_name}/{img_stem}.jpg

Split ratios: 70% train, 15% val, 15% test (random split by image).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import CLASSIFIER_DATASET_DIR, TRAFFIC_CLASSES, PREPROCESSING, VISDRONE_CLASS_MAP, VISDRONE_IGNORE_CLASSES

random.seed(42)


def build_classifier_dataset(raw_dir: Path, output_dir: Path):
    """
    Extract vehicle/pedestrian crops for CNN/SVM classifier training.

    Args:
        raw_dir: Path to extracted VisDrone dataset.
        output_dir: Path to save cropped images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (raw_dir / "VisDrone2019-DET-train" / "images", raw_dir / "VisDrone2019-DET-train" / "annotations"),
        "val": (raw_dir / "VisDrone2019-DET-val" / "images", raw_dir / "VisDrone2019-DET-val" / "annotations"),
        "test": (raw_dir / "VisDrone2019-DET-test-dev" / "images", raw_dir / "VisDrone2019-DET-test-dev" / "annotations"),
    }

    class_samples = {cls: [] for cls in TRAFFIC_CLASSES}

    for split_name, (img_dir, ann_dir) in splits.items():
        if not img_dir.exists() or not ann_dir.exists():
            continue

        for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc=f"Scanning {split_name}"):
            ann_path = ann_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            with open(ann_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue

                    bbox_left = int(parts[0])
                    bbox_top = int(parts[1])
                    bbox_width = int(parts[2])
                    bbox_height = int(parts[3])
                    score = int(parts[4])
                    category = int(parts[5])

                    if category in VISDRONE_IGNORE_CLASSES or category not in VISDRONE_CLASS_MAP:
                        continue
                    if bbox_width <= 0 or bbox_height <= 0:
                        continue

                    yolo_class = VISDRONE_CLASS_MAP[category]
                    class_name = TRAFFIC_CLASSES[yolo_class]

                    # Add 10% padding
                    pad_x = int(bbox_width * 0.1)
                    pad_y = int(bbox_height * 0.1)

                    x1 = max(0, bbox_left - pad_x)
                    y1 = max(0, bbox_top - pad_y)
                    x2 = min(w, bbox_left + bbox_width + pad_x)
                    y2 = min(h, bbox_top + bbox_height + pad_y)

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    size = PREPROCESSING["classifier_input_size"]
                    crop_resized = cv2.resize(crop, (size, size))

                    class_samples[class_name].append({
                        "crop": crop_resized,
                        "split": split_name,
                        "stem": f"{img_path.stem}_{x1}_{y1}",
                    })

    # Save crops organized by split/class
    for class_name, samples in class_samples.items():
        for sample in samples:
            split_dir = output_dir / sample["split"] / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / f"{sample['stem']}.jpg"
            cv2.imwrite(str(out_path), sample["crop"])

    print(f"\nClassifier dataset built at: {output_dir}")
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if split_dir.exists():
            total = sum(1 for _ in split_dir.rglob("*.jpg"))
            print(f"  {split}: {total} images")


def main():
    parser = argparse.ArgumentParser(description="Build classifier dataset from VisDrone")
    parser.add_argument("--raw", default="data/raw/visdrone",
                        help="Path to extracted VisDrone dataset")
    parser.add_argument("--output", default=str(CLASSIFIER_DATASET_DIR),
                        help="Output classifier dataset directory")
    args = parser.parse_args()

    build_classifier_dataset(Path(args.raw), Path(args.output))


if __name__ == "__main__":
    main()
