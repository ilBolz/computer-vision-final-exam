"""
SVM-based vehicle classifier for traffic monitoring.

Wraps scikit-learn SVM + StandardScaler for classifying vehicle crops
using HOG features extracted from image patches.
"""

import pickle

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from src.config import CLASSICAL


class VehicleClassifier:
    """SVM classifier for vehicle type recognition using HOG features."""

    def __init__(self, kernel=None, C=None):
        """
        Initialize classifier.

        Args:
            kernel: SVM kernel type (default from config).
            C: Regularization parameter (default from config).
        """
        self.kernel = kernel or CLASSICAL.get("svm_kernel", "rbf")
        self.C = C or CLASSICAL.get("svm_c", 1.0)
        self.scaler = StandardScaler()
        self.model = SVC(kernel=self.kernel, C=self.C, probability=True)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the SVM classifier.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label vector (n_samples,).
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, features: np.ndarray):
        """
        Predict class and confidence for a single sample or batch.

        Args:
            features: Feature vector (n_features,) or matrix (n_samples, n_features).

        Returns:
            For single sample: (class_index, confidence).
            For batch: (class_indices, confidences).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet.")

        is_single = features.ndim == 1
        if is_single:
            features = features.reshape(1, -1)

        X_scaled = self.scaler.transform(features)
        probs = self.model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        if is_single:
            return int(preds[0]), float(confs[0])
        return preds.astype(int).tolist(), confs.tolist()

    def save(self, path: str):
        """Save scaler and model to disk."""
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model}, f)

    def load(self, path: str):
        """Load scaler and model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.is_trained = True
