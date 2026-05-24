"""
Unit tests for vehicle classifiers (CNN).
"""

import numpy as np
import pytest


def test_cnn_output_shape():
    """CNN should output tensor of shape [B, num_classes]."""
    torch = pytest.importorskip("torch")
    from src.deep_learning.vehicle_net import VehicleNet
    from src.config import NUM_TRAFFIC_CLASSES

    model = VehicleNet(num_classes=NUM_TRAFFIC_CLASSES)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, NUM_TRAFFIC_CLASSES)


def test_cnn_probabilities_sum_to_one():
    """After softmax, probabilities should sum to ~1."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from src.deep_learning.vehicle_net import VehicleNet
    from src.config import NUM_TRAFFIC_CLASSES

    model = VehicleNet(num_classes=NUM_TRAFFIC_CLASSES)
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
