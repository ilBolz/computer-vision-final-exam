"""
Entrypoint for training the classical HOG + SVM vehicle classifier.

Wraps src.classical.train with default paths from config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

from src.config import HOG_SVM_PATH, CLASSIFIER_DATASET_DIR
from src.classical.train import train_hog_svm


def main():
    parser = argparse.ArgumentParser(description="Train HOG+SVM vehicle classifier")
    parser.add_argument("--data", default=str(CLASSIFIER_DATASET_DIR / "train"),
                        help="Path to classifier train dataset (ImageFolder)")
    parser.add_argument("--output", default=str(HOG_SVM_PATH),
                        help="Output model path")
    args = parser.parse_args()

    train_hog_svm(Path(args.data), Path(args.output))
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
