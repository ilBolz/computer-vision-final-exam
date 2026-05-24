"""
Training pipeline for classical HOG + SVM vehicle classifier.

Extracts HOG features from vehicle crops and trains an SVM with
StandardScaler normalization.
"""

from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.classical.hog_detector import HOGDetector
from src.classical.vehicle_classifier import VehicleClassifier
from src.config import CLASSICAL, TRAFFIC_CLASSES


def extract_hog_features_from_dataset(dataset_dir: Path):
    """
    Extract HOG features from an ImageFolder-style dataset.

    Args:
        dataset_dir: Root directory with subfolders named by class.

    Returns:
        X: Feature matrix (n_samples, n_features).
        y: Label vector (n_samples,).
        class_names: Ordered list of class names matching label indices.
    """
    hog = HOGDetector(use_default_detector=False)
    features = []
    labels = []
    class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_to_idx[class_dir.name]
        for img_path in class_dir.glob("*.jpg"):
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            # Resize to HOG window size
            win_w, win_h = CLASSICAL["hog_win_size"]
            image = cv2.resize(image, (win_w, win_h))
            desc = hog.compute_descriptor(image)
            features.append(desc.flatten())
            labels.append(label)

    return np.array(features), np.array(labels), class_names


def train_hog_svm(dataset_dir: str, output_path: str = None, test_size: float = 0.2):
    """
    Train HOG + SVM classifier on a dataset of vehicle crops.

    Args:
        dataset_dir: Path to dataset root (ImageFolder structure).
        output_path: Path to save trained model.
        test_size: Fraction of data to use for validation.

    Returns:
        Trained VehicleClassifier instance.
    """
    dataset_dir = Path(dataset_dir)
    print(f"Loading dataset from {dataset_dir}...")
    X, y, class_names = extract_hog_features_from_dataset(dataset_dir)
    print(f"Loaded {len(y)} samples, {len(class_names)} classes: {class_names}")

    if len(y) == 0:
        raise ValueError("No samples found in dataset.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    classifier = VehicleClassifier()
    print("Training SVM...")
    classifier.train(X_train, y_train)

    # Validation accuracy
    val_preds, _ = classifier.predict(X_val)
    accuracy = np.mean(val_preds == y_val)
    print(f"Validation accuracy: {accuracy:.4f}")

    if output_path:
        classifier.save(output_path)
        print(f"Model saved to {output_path}")

    return classifier
