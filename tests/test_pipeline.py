"""
End-to-end pipeline smoke tests for Traffic Monitoring.

Tests that all modules import correctly and basic operations
work on synthetic data.
"""

import numpy as np
import pytest


class TestPreprocessing:
    def test_resize_image(self):
        from src.preprocessing.image_pipeline import resize_image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = resize_image(img, (320, 240))
        assert out.shape == (240, 320, 3)


class TestYOLOTrafficDetector:
    def test_import(self):
        from src.deep_learning.yolo_detector import YOLOTrafficDetector
        assert YOLOTrafficDetector is not None


class TestVehicleNet:
    def test_import(self):
        torch = pytest.importorskip("torch")
        from src.deep_learning.vehicle_net import VehicleNet
        assert VehicleNet is not None


class TestSimpleTracker:
    def test_import(self):
        from src.postprocessing.tracker import SimpleTracker
        assert SimpleTracker is not None

    def test_update_and_count(self):
        from src.postprocessing.tracker import SimpleTracker
        tracker = SimpleTracker()
        detections = [
            [2, 0.9, 100, 100, 50, 50],   # car
            [4, 0.8, 200, 200, 60, 60],   # truck
        ]
        tracks = tracker.update(detections)
        assert len(tracks) == 2
        counts = tracker.check_crossings(tracks, "horizontal", 0.5, 480, 640)
        # No crossing because centroids are above line
        assert sum(counts.values()) == 0


class TestWebcamStreamer:
    def test_import(self):
        from src.webcam.streamer import WebcamStreamer
        assert WebcamStreamer is not None


class TestTrafficPipeline:
    def test_import(self):
        from src.webcam.pipeline import TrafficPipeline
        assert TrafficPipeline is not None

    def test_counts_initialized(self):
        from src.webcam.pipeline import TrafficPipeline
        from src.config import TRAFFIC_CLASSES
        pipeline = TrafficPipeline()
        assert len(pipeline.counts) == len(TRAFFIC_CLASSES)
        for v in pipeline.counts.values():
            assert v == 0
