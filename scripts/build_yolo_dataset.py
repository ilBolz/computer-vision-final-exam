"""
Convert VisDrone2019-DET annotations to YOLO format for traffic detection.

Creates multi-class dataset (7 traffic classes) with normalized bbox labels.
Output structure:
    data/processed/yolo/{train,val,test}/images
    data/processed/yolo/{train,val,test}/labels

Filtered classes:
    0: pedestrian, 1: bicycle, 2: car, 3: van, 4: truck, 5: bus, 6: motor
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import TRAFFIC_CLASSES, YOLO_DATASET_DIR
from src.preprocessing.yolo_formatter import convert_visdrone_to_yolo

random.seed(42)


def build_yolo_dataset(raw_dir: Path, output_dir: Path):
    """
    Build YOLO-formatted dataset from VisDrone raw data.

    Args:
        raw_dir: Path to extracted VisDrone dataset (e.g., data/raw/visdrone).
        output_dir: Path to save YOLO dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (raw_dir / "VisDrone2019-DET-train" / "images", raw_dir / "VisDrone2019-DET-train" / "annotations"),
        "val": (raw_dir / "VisDrone2019-DET-val" / "images", raw_dir / "VisDrone2019-DET-val" / "annotations"),
        "test": (raw_dir / "VisDrone2019-DET-test-dev" / "images", raw_dir / "VisDrone2019-DET-test-dev" / "annotations"),
    }

    for split_name, (img_dir, ann_dir) in splits.items():
        if not img_dir.exists() or not ann_dir.exists():
            print(f"WARNING: Split '{split_name}' not found at expected paths. Skipping.")
            continue

        img_out_dir = output_dir / split_name / "images"
        lbl_out_dir = output_dir / split_name / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(img_dir.glob("*.jpg"))
        for img_path in tqdm(img_files, desc=f"YOLO/{split_name}"):
            ann_path = ann_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            shutil.copy2(str(img_path), str(img_out_dir / img_path.name))
            convert_visdrone_to_yolo(ann_path, w, h, lbl_out_dir / (img_path.stem + ".txt"))

    # Write dataset YAML config for Ultralytics
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("\n")
        f.write(f"nc: {len(TRAFFIC_CLASSES)}\n")
        f.write(f"names: {TRAFFIC_CLASSES}\n")

    print(f"\nYOLO dataset built at: {output_dir}")
    for split in ["train", "val", "test"]:
        n = len(list((output_dir / split / "images").glob("*.jpg")))
        print(f"  {split}: {n} images")
    print(f"  YAML config: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Build YOLO dataset from VisDrone")
    parser.add_argument("--raw", default="data/raw/visdrone",
                        help="Path to extracted VisDrone dataset")
    parser.add_argument("--output", default=str(YOLO_DATASET_DIR),
                        help="Output YOLO dataset directory")
    args = parser.parse_args()

    build_yolo_dataset(Path(args.raw), Path(args.output))


if __name__ == "__main__":
    main()
