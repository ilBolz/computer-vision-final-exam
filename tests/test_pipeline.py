"""
End-to-end pipeline smoke tests for Traffic Monitoring.

Tests that all modules import correctly and basic operations
work on synthetic data.
"""

import numpy as np
import pytest

from src.preprocessing.image_pipeline import preprocess_for_yolo, preprocess_for_classical


class TestPreprocessing:
    def test_yolo_blob_shape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob = preprocess_for_yolo(img)
        assert blob.shape == (1, 3, 416, 416)

    def test_classical_grayscale(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = preprocess_for_classical(img)
        assert len(out.shape) == 2
        assert out.shape == (480, 640)


class TestHOGDetector:
    def test_import(self):
        from src.classical.hog_detector import HOGDetector
        assert HOGDetector is not None

    def test_instantiation(self):
        from src.classical.hog_detector import HOGDetector
        det = HOGDetector(use_default_detector=True)
        assert det is not None


class TestVehicleClassifier:
    def test_import(self):
        from src.classical.vehicle_classifier import VehicleClassifier
        assert VehicleClassifier is not None


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

    def test_mode_switch(self):
        from src.webcam.pipeline import TrafficPipeline
        pipeline = TrafficPipeline(mode="classical")
        assert pipeline.mode == "classical"
        pipeline.switch_mode("deep")
        assert pipeline.mode == "deep"

    def test_counts_initialized(self):
        from src.webcam.pipeline import TrafficPipeline
        from src.config import TRAFFIC_CLASSES
        pipeline = TrafficPipeline(mode="deep")
        assert len(pipeline.counts) == len(TRAFFIC_CLASSES)
        for v in pipeline.counts.values():
            assert v == 0
