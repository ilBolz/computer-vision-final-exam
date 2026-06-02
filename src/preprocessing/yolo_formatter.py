"""Convert VisDrone annotations to YOLO format."""

import shutil
from pathlib import Path

from src.config import VISDRONE_CLASS_MAP, VISDRONE_IGNORE_CLASSES


def convert_visdrone_to_yolo(visdrone_txt_path: Path, img_width: int, img_height: int, output_txt_path: Path):
    """Convert one VisDrone txt file to YOLO txt."""
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

            x_center = (bbox_left + bbox_width / 2.0) / img_width
            y_center = (bbox_top + bbox_height / 2.0) / img_height
            w_norm = bbox_width / img_width
            h_norm = bbox_height / img_height

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(yolo_lines))



