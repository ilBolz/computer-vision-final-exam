"""
Entrypoint for training the PyTorch vehicle classifier.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

from src.config import CLASSIFIER_DATASET_DIR, VEHICLE_CNN_PATH
from src.deep_learning.train_classifier import train_classifier


def main():
    parser = argparse.ArgumentParser(description="Train vehicle CNN classifier")
    parser.add_argument("--data", default=str(CLASSIFIER_DATASET_DIR),
                        help="Path to classifier dataset directory")
    parser.add_argument("--output", default=str(VEHICLE_CNN_PATH),
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_classifier(Path(args.data), Path(args.output), args.epochs)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
