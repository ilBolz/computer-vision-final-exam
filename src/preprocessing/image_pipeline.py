"""
Image preprocessing pipeline for traffic monitoring.

Encapsulates resize and normalization logic in a modular, reusable interface.
"""

import cv2
import numpy as np


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image to target dimensions.

    Args:
        image: Input image.
        target_size: (width, height).

    Returns:
        Resized image.
    """
    return cv2.resize(image, target_size)
