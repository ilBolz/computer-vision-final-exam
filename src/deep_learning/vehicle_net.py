"""
PyTorch CNN / ResNet18 classifier for vehicle type recognition.

Architecture options:
    - resnet18: Pre-trained on ImageNet, FC replaced for 7-class traffic.
    - custom: Lightweight CNN for CPU inference.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import CNN


class VehicleNet(nn.Module):
    """Vehicle classification network."""

    def __init__(self, num_classes=None, architecture=None):
        """
        Initialize network.

        Args:
            num_classes: Number of output classes (default from config).
            architecture: 'resnet18' or 'custom'.
        """
        super().__init__()
        self.num_classes = num_classes or CNN.get("num_classes", 7)
        self.architecture = architecture or CNN.get("architecture", "resnet18")

        if self.architecture == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif self.architecture == "custom":
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.num_classes),
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def forward(self, x):
        """Forward pass."""
        return self.model(x)


def get_model(num_classes=None, architecture=None):
    """
    Factory function to create a VehicleNet model.

    Args:
        num_classes: Number of output classes.
        architecture: 'resnet18' or 'custom'.

    Returns:
        VehicleNet instance.
    """
    return VehicleNet(num_classes=num_classes, architecture=architecture)
