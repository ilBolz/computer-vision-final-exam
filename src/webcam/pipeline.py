"""
Live webcam pipeline for real-time traffic monitoring with YOLO + CNN.

Supports:
    - DEEP mode: YOLOv8n multi-class detection + optional ResNet18 refinement

Controls:
    - 's' -> screenshot
    - 'q' -> quit
"""

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.deep_learning.yolo_detector import YOLOTrafficDetector
from src.deep_learning.vehicle_net import get_model
from src.postprocessing.tracker import SimpleTracker
from src.utils.visualization import draw_detections, draw_count_line, draw_counters
from src.config import (
    CNN,
    PREPROCESSING,
    TRAFFIC_CLASSES,
    VEHICLE_CNN_PATH,
    VISUALIZATION,
    WEBCAM,
)


class TrafficPipeline:
    """Live traffic monitoring pipeline with vehicle counting."""

    def __init__(self):
        """Initialize pipeline."""
        self.yolo = None
        self.cnn = None
        self.cnn_transform = None
        self.tracker = SimpleTracker()
        self.counts = {cls: 0 for cls in range(len(TRAFFIC_CLASSES))}
        self._load_models()

    def _load_models(self):
        """Load deep learning models."""
        self.yolo = YOLOTrafficDetector()
        self.cnn = get_model(num_classes=CNN["num_classes"])
        if VEHICLE_CNN_PATH.exists():
            self.cnn.load_state_dict(
                torch.load(VEHICLE_CNN_PATH, map_location="cpu")
            )
        else:
            print("WARNING: Vehicle CNN model not found. Run train_classifier.py first.")
        self.cnn.eval()

        size = PREPROCESSING["classifier_input_size"]
        mean = PREPROCESSING["normalize_mean"]
        std = PREPROCESSING["normalize_std"]
        self.cnn_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Args:
            frame: BGR image from webcam.

        Returns:
            Annotated frame with detections and counts.
        """
        annotated = frame.copy()
        start_time = time.perf_counter()

        detections = self.yolo.detect(frame)

        # Optional: refine with CNN classifier on crops
        if self.cnn is not None and VEHICLE_CNN_PATH.exists():
            refined = []
            for det in detections:
                cls_id, conf, x, y, w, h = det
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    tensor = self.cnn_transform(crop).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.cnn(tensor)
                        probs = F.softmax(logits, dim=1)
                        pred = int(torch.argmax(probs, dim=1)[0])
                        conf_cls = float(probs[0][pred])
                    refined.append([pred, conf_cls, x, y, w, h])
                else:
                    refined.append(det)
            detections = refined

        # Update tracker and counts
        tracks = self.tracker.update(detections)
        img_h, img_w = frame.shape[:2]
        crossings = self.tracker.check_crossings(
            tracks,
            line_orientation=WEBCAM.get("count_line_orientation", "horizontal"),
            line_ratio=WEBCAM.get("count_line_y_ratio", 0.5),
            img_h=img_h,
            img_w=img_w,
        )
        for cls_id, inc in crossings.items():
            self.counts[cls_id] = self.counts.get(cls_id, 0) + inc

        annotated = draw_detections(frame, detections, class_names=TRAFFIC_CLASSES)

        # Draw count line
        annotated = draw_count_line(
            annotated,
            orientation=WEBCAM.get("count_line_orientation", "horizontal"),
            ratio=WEBCAM.get("count_line_y_ratio", 0.5),
        )

        # Draw counters
        annotated = draw_counters(annotated, self.counts)

        # Draw FPS
        fps = 1.0 / (time.perf_counter() - start_time + 1e-6)
        annotated = self._draw_hud(annotated, fps)

        return annotated

    def _draw_hud(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS indicator."""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )
        return frame
