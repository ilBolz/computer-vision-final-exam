"""
Evaluation script for the Deep Learning traffic pipeline.

Evaluates YOLO + CNN on test images / datasets and generates
a report with detection metrics, classification accuracy,
and inference speed.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import TRAFFIC_CLASSES, YOLO_TRAFFIC_MODEL_PATH, VEHICLE_CNN_PATH
from src.deep_learning.yolo_detector import YOLOTrafficDetector
from src.deep_learning.vehicle_net import VehicleNet
from src.evaluation.benchmark import benchmark_detector
from src.evaluation.metrics import plot_confusion_matrix
from src.utils.visualization import draw_detections


class DeepPipeline:
    """End-to-end deep learning pipeline: YOLO -> crop -> CNN."""

    def __init__(self, device=None):
        self.detector = YOLOTrafficDetector(str(YOLO_TRAFFIC_MODEL_PATH))
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.classifier = VehicleNet(num_classes=len(TRAFFIC_CLASSES))
        if VEHICLE_CNN_PATH.exists():
            self.classifier.load_state_dict(torch.load(VEHICLE_CNN_PATH, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        self.img_size = 224

    def detect(self, image: np.ndarray):
        """Return detections in [class_id, confidence, x, y, w, h] format."""
        import torchvision.transforms as T

        dets = self.detector.detect(image)
        refined = []

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for det in dets:
            _, conf_det, x, y, w, h = det
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            tensor = transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_class = int(torch.argmax(probs))
                conf = float(probs[pred_class])

            refined.append([pred_class, conf * conf_det, x, y, w, h])
        return refined


def evaluate_on_images(pipeline: DeepPipeline, image_paths, output_dir: Path):
    """Run pipeline on a list of images and collect detection + timing metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    valid_paths = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
            valid_paths.append(p)

    if len(images) == 0:
        raise ValueError("No valid images provided for evaluation.")

    print("Benchmarking deep learning pipeline...")
    bench = benchmark_detector(pipeline, images, warmup=1)

    results = []

    for i, image in enumerate(images):
        dets = pipeline.detect(image)
        results.append(dets)

        vis = draw_detections(image.copy(), dets, class_names=TRAFFIC_CLASSES)
        cv2.imwrite(str(output_dir / f"deep_{i:03d}.jpg"), vis)

    report = {
        "num_images": len(images),
        "deep": {
            "fps": bench["mean_fps"],
            "latency_ms": bench["mean_latency_ms"],
            "total_detections": sum(len(r) for r in results),
        },
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation complete. Results saved to {output_dir}")
    return report


def evaluate_classifier_dataset(pipeline: DeepPipeline, data_dir: Path, output_dir: Path):
    """Evaluate pipeline on the classifier crop dataset."""
    from torchvision import datasets
    import torchvision.transforms as T

    output_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    y_true = []
    y_pred = []

    print(f"Evaluating on {len(dataset)} samples from {data_dir}...")

    for idx in range(len(dataset)):
        tensor, label = dataset[idx]
        img_path = dataset.samples[idx][0]
        image = cv2.imread(img_path)
        if image is None:
            continue

        y_true.append(label)

        dets = pipeline.detect(image)
        if len(dets) > 0:
            dets.sort(key=lambda d: d[1], reverse=True)
            y_pred.append(dets[0][0])
        else:
            y_pred.append(-1)

    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]

    def _metrics(y_t, y_p):
        return {
            "accuracy": accuracy_score(y_t, y_p),
            "precision": precision_score(y_t, y_p, average="macro", zero_division=0),
            "recall": recall_score(y_t, y_p, average="macro", zero_division=0),
            "f1": f1_score(y_t, y_p, average="macro", zero_division=0),
        }

    metrics = {
        "num_samples": len(y_true),
        "deep": {
            **_metrics([v[0] for v in valid], [v[1] for v in valid]),
            "detection_rate": len(valid) / len(y_true),
        },
    }

    with open(output_dir / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if len(valid) > 0:
        plot_confusion_matrix(
            [v[0] for v in valid],
            [v[1] for v in valid],
            TRAFFIC_CLASSES,
            save_path=output_dir / "confusion_deep.jpg",
        )

    print("Classification evaluation complete.")
    print(f"  Deep accuracy:      {metrics['deep']['accuracy']:.4f} (detected {metrics['deep']['detection_rate']:.2%})")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Deep Learning traffic pipeline")
    parser.add_argument("--mode", choices=["images", "dataset"], default="images",
                        help="Evaluation mode: 'images' for raw images, 'dataset' for classifier test set")
    parser.add_argument("--images", nargs="*", help="Paths to test images (mode=images)")
    parser.add_argument("--dataset", default="data/processed/classifier/test",
                        help="Path to classifier test dataset (mode=dataset)")
    parser.add_argument("--output", default="docs/results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading deep learning pipeline...")
    pipeline = DeepPipeline()

    if args.mode == "images":
        if not args.images:
            raise ValueError("--images required when mode=images")
        image_paths = [Path(p) for p in args.images]
        report = evaluate_on_images(pipeline, image_paths, output_dir)
        print("\n--- Detection Evaluation Report ---")
        print(f"Images processed: {report['num_images']}")
        print(f"Deep FPS:      {report['deep']['fps']:.2f}  |  Detections: {report['deep']['total_detections']}")
    else:
        metrics = evaluate_classifier_dataset(pipeline, Path(args.dataset), output_dir)
        print("\n--- Classification Evaluation Report ---")
        print(f"Deep:      accuracy={metrics['deep']['accuracy']:.4f}, precision={metrics['deep']['precision']:.4f}, "
              f"recall={metrics['deep']['recall']:.4f}, f1={metrics['deep']['f1']:.4f}")


if __name__ == "__main__":
    main()
