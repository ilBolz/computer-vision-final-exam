"""Live webcam pipeline: YOLO detection + tracker + counting."""

import time
import cv2
import numpy as np

from src.deep_learning.yolo_detector import YOLOTrafficDetector
from src.postprocessing.tracker import SimpleTracker
from src.utils.visualization import draw_detections, draw_count_line, draw_counters
from src.config import (
    TRAFFIC_CLASSES,
    VISUALIZATION,
    WEBCAM,
)


class TrafficPipeline:
    def __init__(self):
        self.yolo = YOLOTrafficDetector()
        self.tracker = SimpleTracker()
        self.counts = {cls: 0 for cls in range(len(TRAFFIC_CLASSES))}

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame and return annotated output."""
        annotated = frame.copy()
        start_time = time.perf_counter()

        detections = self.yolo.detect(frame)
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
        annotated = draw_count_line(
            annotated,
            orientation=WEBCAM.get("count_line_orientation", "horizontal"),
            ratio=WEBCAM.get("count_line_y_ratio", 0.5),
        )
        annotated = draw_counters(annotated, self.counts)

        fps = 1.0 / (time.perf_counter() - start_time + 1e-6)
        annotated = self._draw_hud(annotated, fps)

        return annotated

    def _draw_hud(self, frame: np.ndarray, fps: float) -> np.ndarray:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )
        return frame
