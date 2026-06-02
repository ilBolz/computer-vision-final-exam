"""Fine-tune YOLOv8n on VisDrone traffic data."""

import argparse
from pathlib import Path

from ultralytics import YOLO as YOLOModel

from src.config import YOLO


def train_yolo(data_yaml: Path, output_dir: Path, epochs: int = None):
    """Train YOLOv8n."""
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



