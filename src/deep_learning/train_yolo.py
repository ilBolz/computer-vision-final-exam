"""
Fine-tuning script for YOLOv8n traffic detection.

Trains YOLOv8n on VisDrone2019-DET multi-class vehicle dataset.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO as YOLOModel

from src.config import YOLO


def train_yolo(data_yaml: Path, output_dir: Path, epochs: int = None):
    """
    Fine-tune YOLOv8n on traffic detection dataset.

    Args:
        data_yaml: Path to YOLO dataset YAML config.
        output_dir: Directory to save training outputs.
        epochs: Number of training epochs.
    """
    epochs = epochs or YOLO["epochs"]

    model = YOLOModel("yolov8n.pt")

    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=YOLO["imgsz"],
        batch=YOLO["batch"],
        lr0=YOLO["lr0"],
        project=str(output_dir),
        name="traffic_detector",
        exist_ok=True,
        device=device,
    )

    best_path = Path(output_dir) / "traffic_detector" / "weights" / "best.pt"
    print(f"Best model saved to: {best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n traffic detector")
    parser.add_argument("--data", default="data/processed/yolo/data.yaml",
                        help="Path to YOLO dataset YAML")
    parser.add_argument("--output", default="models/yolov8n_traffic",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_yolo(Path(args.data), Path(args.output), args.epochs)


if __name__ == "__main__":
    main()
