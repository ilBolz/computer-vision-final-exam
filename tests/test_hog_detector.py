"""
Unit tests for HOG detector and descriptor computation.
"""

import numpy as np
import pytest
import cv2

from src.config import CLASSICAL


def test_hog_descriptor_shape():
    """HOG descriptor should have consistent dimensions for a fixed window."""
    from src.classical.hog_detector import HOGDetector

    hog = HOGDetector(use_default_detector=False)
    win_w, win_h = CLASSICAL["hog_win_size"]
    gray = np.random.randint(0, 255, (win_h, win_w), dtype=np.uint8)
    desc = hog.compute_descriptor(gray)
    assert desc.ndim == 1
    assert desc.shape[0] > 0


def test_hog_detector_default_detects():
    """Default HOG detector should return a list on synthetic data."""
    from src.classical.hog_detector import HOGDetector

    detector = HOGDetector(use_default_detector=True)
    # Create a synthetic image that resembles a pedestrian shape
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a vertical rectangle to simulate a person
    cv2.rectangle(img, (300, 100), (340, 380), (128, 128, 128), -1)

    dets = detector.detect(img)
    assert isinstance(dets, list)
    # Default detector may or may not detect on synthetic data,
    # but the output format must be consistent when detections exist
    for det in dets:
        assert len(det) == 6
        cls_id, conf, x, y, w, h = det
        assert isinstance(cls_id, int)
        assert isinstance(conf, float)
        assert w > 0 and h > 0


def test_hog_sliding_window_placeholder():
    """Custom sliding window detection placeholder should return empty list."""
    from src.classical.hog_detector import HOGDetector

    detector = HOGDetector(use_default_detector=False)
    gray = np.zeros((480, 640), dtype=np.uint8)
    dets = detector._sliding_window_detect(gray, (8, 8), 1.05)
    assert dets == []
