"""
Entrypoint for training YOLOv8n traffic detector.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

from src.config import YOLO_DATASET_DIR
from src.deep_learning.train_yolo import train_yolo


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n traffic detector")
    parser.add_argument("--data", default=str(YOLO_DATASET_DIR / "data.yaml"),
                        help="Path to YOLO dataset YAML")
    parser.add_argument("--output", default="models/yolov8n_traffic",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    best = train_yolo(Path(args.data), Path(args.output), args.epochs)
    print(f"Best model: {best}")


if __name__ == "__main__":
    main()
