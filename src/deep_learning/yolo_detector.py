"""YOLOv8n wrapper for traffic detection."""

from pathlib import Path

import numpy as np
import cv2

from src.config import YOLO, YOLO_TRAFFIC_MODEL_PATH, YOLO_TRAFFIC_MODEL_FALLBACK


class YOLOTrafficDetector:
    def __init__(self, model_path=None, conf_threshold=None, nms_threshold=None):
        """Load YOLO model with fallback to COCO weights."""
        self.conf_threshold = conf_threshold or YOLO.get("conf_threshold", 0.5)
        self.nms_threshold = nms_threshold or YOLO.get("nms_threshold", 0.4)

        from ultralytics import YOLO as YOLOModel

        model_path = self._resolve_model_path(model_path)
        self.model = YOLOModel(model_path)

    def _resolve_model_path(self, model_path):
        if model_path is not None and Path(model_path).exists():
            return model_path
        if YOLO_TRAFFIC_MODEL_PATH.exists():
            return str(YOLO_TRAFFIC_MODEL_PATH)
        if YOLO_TRAFFIC_MODEL_FALLBACK.exists():
            return str(YOLO_TRAFFIC_MODEL_FALLBACK)
        return YOLO.get("model_name", "yolov8n.pt")

    def detect(self, image: np.ndarray) -> list:
        """Run detection and return [class_id, conf, x, y, w, h] in absolute xywh."""
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            verbose=False,
        )

        detections = []
        if len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        img_h, img_w = image.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            detections.append([cls_id, conf, float(x), float(y), float(w), float(h)])

        return detections
