# Real-Time Traffic Monitoring Framework

A modular computer vision system using deep learning (YOLOv8n multi-class detection + ResNet18 vehicle classification) for real-time traffic monitoring and vehicle counting.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

This project implements a complete traffic monitoring pipeline for 7 vehicle/pedestrian classes using deep learning:

| Stage | Deep Learning (YOLOv8 + CNN) |
|-------|------------------------------|
| Preprocessing | Resize 640×640, normalization |
| Features | Learned CNN features (ResNet18) |
| Detection | YOLOv8n multi-class detector |
| Classification | ResNet18 fine-tuned classifier |
| Post-processing | NMS + IOU tracker + counting line |
| Evaluation | mAP@0.5, Accuracy, FPS |

### Design Principles
- **Modularity** — Swap backbones without touching visualization or metrics logic
- **Explainability** — Custom IoU and NMS implementations with mathematical transparency
- **Real-time** — Webcam pipeline with vehicle counting and HUD overlay

---

## Quick Start

### 1. Setup Environment

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate cv-exam

# Option B: venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download VisDrone2019-DET (~1.5GB) — free, no registration required
python scripts/download_visdrone.py --output data/raw/visdrone
```

### 3. Build Processed Datasets

```bash
# YOLO format for traffic detection (7 classes)
python scripts/build_yolo_dataset.py

# Vehicle/pedestrian crops for CNN classifier
python scripts/build_classifier_dataset.py
```

### 4. Train Models

```bash
# Deep Learning: YOLOv8n traffic detector
python scripts/train_yolo.py --epochs 20

# Deep Learning: ResNet18 vehicle classifier
python scripts/train_classifier.py --epochs 20
```

### 5. Run Inference

```bash
# Evaluation on test images
python -m src.evaluation.compare --mode images --images img1.jpg img2.jpg --output docs/results/

# Evaluation on classifier test dataset
python -m src.evaluation.compare --mode dataset --output docs/results/
```

### 6. Live Webcam / Video Demo

```bash
# Deep mode (YOLO + CNN, GPU recommended)
python scripts/run_webcam.py --source 0

# On a video file
python scripts/run_webcam.py --source path/to/video.mp4

# Controls:
#   's' → screenshot
#   'q' → quit
```

---

## Repository Structure

```
computer_vision_exam/
├── data/
│   ├── raw/                   # VisDrone subset (gitignored)
│   ├── processed/
│   │   ├── yolo/              # YOLO format dataset
│   │   └── classifier/        # 224×224 vehicle crops
│   └── annotations/           # JSON splits
├── models/
│   ├── vehicle_cnn.pt         # ResNet18 weights
│   └── yolov8n_traffic.pt     # Fine-tuned YOLO (gitignored)
├── src/
│   ├── config.py              # Centralized configuration
│   ├── preprocessing/         # Image pipelines, YOLO formatter
│   ├── deep_learning/         # YOLOv8 + CNN pipeline
│   ├── postprocessing/        # IoU, NMS, Tracker
│   ├── evaluation/            # Metrics, benchmark, compare
│   ├── webcam/                # Streamer + live pipeline
│   └── utils/                 # Visualization, logging
├── tests/                     # pytest suite
├── scripts/                   # Entrypoints
├── notebooks/                 # Jupyter demos
├── docs/
│   ├── technical_analysis.md  # Source for PDF deliverable
│   └── results/               # Comparison reports (gitignored)
├── requirements.txt
├── environment.yml
├── PROJECT_PLAN.md            # Implementation roadmap
└── README.md
```

---

## Dataset: VisDrone2019-DET

**Source:** https://github.com/VisDrone/VisDrone-Dataset  
**License:** Free for academic/research use  
**Size:** ~10,209 images (train/val/test-dev)  
**Annotations:** Bounding boxes in `[x, y, w, h, score, category, truncation, occlusion]` format

### Filtered Traffic Classes (7 categories)

| ID | Class | Original VisDrone ID |
|----|-------|---------------------|
| 0 | pedestrian | 1 |
| 1 | bicycle | 3 |
| 2 | car | 4 |
| 3 | van | 5 |
| 4 | truck | 6 |
| 5 | bus | 9 |
| 6 | motor | 10 |

Ignored original classes: `ignored regions` (0), `people` (2), `tricycle` (7), `awning-tricycle` (8), `others` (11).

---

## Results

### Deep Learning Pipeline (YOLOv8 + ResNet18)

| Metric | Value |
|--------|-------|
| CNN Validation Accuracy | **TBD** |
| YOLO mAP@0.5 | **TBD** |
| YOLO mAP@0.5:0.95 | **TBD** |
| Inference Time | ~15-30 ms/frame (GPU) |

*Results on VisDrone2019-DET test set (7 classes: pedestrian, bicycle, car, van, truck, bus, motor).*

### End-to-End Evaluation

Run evaluation:

```bash
python -m src.evaluation.compare --mode dataset --output docs/results/
```

Generates:
- `evaluation_report.json` — FPS, latency, total detections
- `classification_metrics.json` — Accuracy, Precision, Recall, F1
- `confusion_deep.jpg` — Confusion matrix

---

## Technical Analysis

The full methodology, experimental results, failure analysis, and ethical considerations are documented in:

- **Source:** `docs/technical_analysis.md`
- **PDF:** `docs/technical_analysis.pdf` (auto-generated from markdown)

---

## Ethical Considerations

- **Privacy** — The webcam pipeline captures video locally. No frames or detections are transmitted to external servers. Obtain user consent before enabling camera access.
- **Bias** — The VisDrone dataset is collected from specific cities in China. Performance may degrade on underrepresented vehicle types, lighting conditions, or road environments.
- **Environmental Impact** — This project uses pre-trained weights (ResNet18, YOLOv8n) to avoid energy-intensive training from scratch. Fine-tuning is limited to 50 epochs.
- **Regulatory Compliance** — Real-time traffic monitoring in public or shared spaces may require consent under GDPR, CCPA, or local privacy laws.

---

## License

[Add your license here]

---

## Acknowledgments

- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset authors (Zhu et al.)
- Ultralytics for YOLOv8
- PyTorch and torchvision teams
