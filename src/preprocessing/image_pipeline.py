"""
Image preprocessing pipeline for traffic monitoring.

Encapsulates cv2.dnn.blobFromImage logic for YOLO inference
and classical HOG preprocessing in a modular, reusable interface.
"""

import cv2
import numpy as np
from src.config import PREPROCESSING, CLASSICAL


def preprocess_for_yolo(image: np.ndarray) -> np.ndarray:
    """
    Transform an input image into a 4D blob suitable for legacy YOLO inference.

    Args:
        image: Input image in BGR format (H x W x C).

    Returns:
        A 4D blob with shape (1, 3, 416, 416).
    """
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=PREPROCESSING["legacy_scalefactor"],
        size=PREPROCESSING["legacy_yolo_input_size"],
        mean=PREPROCESSING["legacy_mean"],
        swapRB=PREPROCESSING["legacy_swap_rb"],
        crop=PREPROCESSING["legacy_crop"],
    )
    return blob


def preprocess_for_classical(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for classical HOG+SVM detector.

    Steps:
        1. Convert to grayscale.
        2. Apply histogram equalization for illumination normalization.

    Args:
        image: Input image in BGR format.

    Returns:
        Grayscale, equalized image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


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


def compute_hog_descriptor(image: np.ndarray) -> np.ndarray:
    """
    Compute HOG descriptor for a grayscale image.

    Args:
        image: Grayscale image. Expected size to match HOG window.

    Returns:
        HOG feature vector (numpy 1D array).
    """
    hog = cv2.HOGDescriptor(
        winSize=CLASSICAL["hog_win_size"],
        blockSize=CLASSICAL["hog_block_size"],
        blockStride=CLASSICAL["hog_block_stride"],
        cellSize=CLASSICAL["hog_cell_size"],
        nbins=CLASSICAL["hog_nbins"],
    )
    return hog.compute(image)
