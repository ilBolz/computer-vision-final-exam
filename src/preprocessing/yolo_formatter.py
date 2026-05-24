"""
Convert VisDrone2019-DET annotations to YOLO format.

VisDrone annotation format per line:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

YOLO format per line:
    <class_id> <x_center> <y_center> <width> <height>

Filtered classes (7 traffic categories):
    pedestrian, bicycle, car, van, truck, bus, motor
"""

import shutil
from pathlib import Path

from src.config import VISDRONE_CLASS_MAP, VISDRONE_IGNORE_CLASSES


def convert_visdrone_to_yolo(visdrone_txt_path: Path, img_width: int, img_height: int, output_txt_path: Path):
    """
    Convert a single VisDrone annotation file to YOLO format.

    Args:
        visdrone_txt_path: Path to VisDrone .txt annotation file.
        img_width: Width of the corresponding image.
        img_height: Height of the corresponding image.
        output_txt_path: Path to write YOLO .txt annotation file.
    """
    yolo_lines = []

    with open(visdrone_txt_path, "r") as f:
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

            if category in VISDRONE_IGNORE_CLASSES:
                continue
            if category not in VISDRONE_CLASS_MAP:
                continue
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            yolo_class = VISDRONE_CLASS_MAP[category]

            # Convert to normalized xywh (center-based)
            x_center = (bbox_left + bbox_width / 2.0) / img_width
            y_center = (bbox_top + bbox_height / 2.0) / img_height
            w_norm = bbox_width / img_width
            h_norm = bbox_height / img_height

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(yolo_lines))


def build_yolo_dataset(visdrone_images_dir: Path, visdrone_annotations_dir: Path, output_dir: Path, splits=None):
    """
    Build a full YOLO dataset from VisDrone raw data.

    Args:
        visdrone_images_dir: Directory containing VisDrone images (e.g., VisDrone2019-DET-train/images).
        visdrone_annotations_dir: Directory containing VisDrone .txt annotations.
        output_dir: Root output directory for YOLO dataset.
        splits: Dictionary {split_name: (image_glob_pattern, annotation_glob_pattern)}.
                If None, assumes a single flat directory and creates a single 'all' split.
    """
    import cv2

    if splits is None:
        splits = {"all": ("*.jpg", "*.txt")}

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (img_glob, ann_glob) in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(visdrone_images_dir.glob(img_glob)):
            ann_path = visdrone_annotations_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                continue

            # Read image dimensions
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            # Copy image
            shutil.copy(str(img_path), str(img_out / img_path.name))

            # Convert annotation
            convert_visdrone_to_yolo(ann_path, w, h, lbl_out / (img_path.stem + ".txt"))

    print(f"YOLO dataset built at {output_dir}")
