"""
Classical HOG-based object detector for traffic monitoring.

Uses OpenCV's HOGDescriptor with a sliding window approach for
vehicle and pedestrian detection. Supports both the default
Dalal-Triggs people detector and custom HOG+SVM classifiers.
"""

import cv2
import numpy as np

from src.config import CLASSICAL


class HOGDetector:
    """HOG descriptor-based detector with multi-scale sliding window."""

    def __init__(self, use_default_detector: bool = True):
        """
        Initialize HOG detector.

        Args:
            use_default_detector: If True, uses OpenCV's pre-trained people detector.
                                  If False, uses custom HOG parameters for training.
        """
        self.use_default = use_default_detector
        self.hog = cv2.HOGDescriptor(
            CLASSICAL["hog_win_size"],
            CLASSICAL["hog_block_size"],
            CLASSICAL["hog_block_stride"],
            CLASSICAL["hog_cell_size"],
            CLASSICAL["hog_nbins"],
        )

        if self.use_default:
            # Use Dalal-Triggs pre-trained people detector
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image: np.ndarray, win_stride=None, padding=None, scale=None):
        """
        Detect objects using HOG + SVM detector.

        Args:
            image: BGR input image.
            win_stride: Window stride (default from config).
            padding: Padding around image (default (8,8)).
            scale: Image pyramid scale (default from config).

        Returns:
            List of detections [class_id, confidence, x, y, w, h].
            For default detector, class_id is always 0 (person-like).
        """
        if win_stride is None:
            win_stride = CLASSICAL.get("stride", (8, 8))
        if padding is None:
            padding = (8, 8)
        if scale is None:
            scale = CLASSICAL.get("scale_factor", 1.05)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.use_default:
            # detectMultiScale returns (rects, weights)
            rects, weights = self.hog.detectMultiScale(
                gray,
                winStride=win_stride,
                padding=padding,
                scale=scale,
            )

            detections = []
            for (x, y, w, h), conf in zip(rects, weights):
                # Normalize confidence roughly to [0,1]
                norm_conf = float(min(abs(conf), 1.0))
                detections.append([0, norm_conf, int(x), int(y), int(w), int(h)])
            return detections
        else:
            # Custom sliding window detection (for trained SVM)
            return self._sliding_window_detect(gray, win_stride, scale)

    def _sliding_window_detect(self, gray: np.ndarray, win_stride, scale):
        """
        Sliding window detection for custom HOG+SVM.

        Note: This is a placeholder for a full implementation.
        In practice, a custom-trained SVM would be loaded and applied
        per window. For the exam project, the default people detector
        or the classifier-based pipeline is recommended.

        Args:
            gray: Grayscale image.
            win_stride: Stride tuple.
            scale: Pyramid scale factor.

        Returns:
            Empty list (placeholder).
        """
        # Placeholder: real implementation would load a custom SVM,
        # slide across scales, and collect positive windows.
        return []

    def compute_descriptor(self, image: np.ndarray) -> np.ndarray:
        """
        Compute HOG descriptor for a single image patch.

        Args:
            image: Grayscale image patch matching win_size.

        Returns:
            HOG feature vector (1D numpy array).
        """
        return self.hog.compute(image)
