# Real-Time Traffic Monitoring

Real-time traffic monitoring system that detects 7 vehicle/pedestrian categories using YOLOv8n. Includes a live webcam pipeline with a virtual counting line.

## What it does

- **Detection**: locates pedestrians, bicycles, cars, vans, trucks, buses and motorcycles in real time
- **Tracking**: keeps object IDs across frames using an IoU-based tracker
- **Counting**: counts vehicles crossing a configurable virtual line

## Tech stack

- Python 3.10+
- OpenCV (video capture, NMS, filtering)
- Ultralytics YOLOv8 (object detection)
- scikit-learn (evaluation metrics)
- pytest (test suite)

## Requirements

- Python 3.10 or newer
- Git
- Webcam (for live mode) or a video file
- ~3 GB free space (dataset + environment + models)
- (Optional but recommended) NVIDIA GPU with CUDA for training and smoother inference

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/computer_vision_exam.git
cd computer_vision_exam
```

### 2. Create the virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify the installation

```bash
pytest
```

All tests should pass.

## Quick start

> **Note**: the very first time you run the webcam pipeline, the YOLO model may take **20-30 seconds** to load. This is normal and depends on your hardware.

### Webcam (built-in — index 0)

```bash
python scripts/run_monitoring.py --source 0
```

### Webcam (external / USB — index 1)

```bash
python scripts/run_monitoring.py --source 1
```

### Video file

A sample video is included in the repository for quick testing:

```bash
python scripts/run_monitoring.py --source demo/traffic_demo.mp4
```

You can also use any of your own videos:

```bash
python scripts/run_monitoring.py --source path/to/video.mp4
```

### Controls while running

- `s` → save screenshot to `docs/results/screenshot.jpg`
- `q` → quit

### Model weights

- **`models/best.pt`** — trained traffic-specific weights (VisDrone, 7 traffic classes). **Included in the repository** so the application works out of the box.
- **`models/yolov8n.pt`** — COCO pre-trained weights (fallback). **Also included**; used automatically only if `best.pt` is missing.

### Native webcam mode

If your webcam freezes or shows black frames with forced resolutions, launch with `--native` to use the camera's default settings:

```bash
python scripts/run_monitoring.py --source 1 --native
```

## Available scripts

| Script                             | Purpose                                             |
| ---------------------------------- | --------------------------------------------------- |
| `scripts/run_monitoring.py`        | Live webcam / video demo (default source 0)         |
| `scripts/benchmark_webcam_open.py` | Benchmark webcam open speed with different backends |
| `scripts/download_visdrone.py`     | Download the VisDrone2019-DET dataset               |
| `scripts/build_yolo_dataset.py`    | Convert annotations to YOLO format                  |
| `scripts/train_yolo.py`            | Train the YOLOv8n traffic detector                  |

## Full pipeline (from scratch)

If you start without models, follow these steps to train them.

### 1. Download the VisDrone2019-DET dataset

```bash
python scripts/download_visdrone.py --output data/raw/visdrone
```

Downloads ~1.5 GB of images and annotations.

### 2. Build the structured datasets

```bash
# YOLO format for the detector
python scripts/build_yolo_dataset.py
```

### 3. Train the models

```bash
# YOLOv8n (~20-30 min on GPU, much longer on CPU)
python scripts/train_yolo.py --epochs 20
```

Outputs are saved under `runs/detect/`. Copy or symlink the resulting `best.pt` to `models/best.pt` if you want it loaded automatically.

### 4. Launch the webcam demo

```bash
python scripts/run_monitoring.py --source 0
```

## Repository structure

```
├── data/
│   ├── raw/               # VisDrone dataset (gitignored)
│   └── processed/         # YOLO dataset (gitignored)
├── models/
│   └── best.pt            # Trained YOLO weights (gitignored)
├── src/
│   ├── config.py          # Centralized configuration
│   ├── preprocessing/     # VisDrone → YOLO format conversion
│   ├── deep_learning/     # YOLO detector & training wrapper
│   ├── postprocessing/    # IoU, tracker, counting line
│   ├── webcam/            # Streamer and live pipeline
│   └── utils/             # Drawing utilities (bboxes, counters)
├── scripts/               # Entrypoints for training, building, webcam, benchmarks
├── tests/                 # pytest suite
├── docs/
│   ├── results/           # Generated screenshots and evaluation plots
│   ├── technical_analysis.md
│   └── technical_analysis.pdf
├── demo/
│   └── traffic_demo.mp4       # Sample video for quick testing
├── requirements.txt
└── README.md
```

## Dataset

**VisDrone2019-DET** — [GitHub](https://github.com/VisDrone/VisDrone-Dataset)

The original dataset has 12 classes. This project filters 7:

| ID  | Class      | Original VisDrone ID |
| --- | ---------- | -------------------- |
| 0   | pedestrian | 1                    |
| 1   | bicycle    | 3                    |
| 2   | car        | 4                    |
| 3   | van        | 5                    |
| 4   | truck      | 6                    |
| 5   | bus        | 9                    |
| 6   | motor      | 10                   |

Ignored classes: `ignored regions`, `people`, `tricycle`, `awning-tricycle`, `others`.

## Notes and troubleshooting

- **First run is slow**: YOLO weights (`yolov8n.pt` or `best.pt`) are loaded at startup. Expect 20-30 seconds before the video window appears.
- **No webcam data leaves the PC**: everything is processed locally.
- **VisDrone is collected in China**: performance may vary in other countries or lighting conditions.

### Windows-specific webcam issues

- **Webcam index 1 (USB) is very slow to open**: this is a known OpenCV behaviour on Windows with the default DirectShow backend. The project already applies the `cv2.CAP_DSHOW` fix automatically in `src/webcam/streamer.py`. If you still experience slowness, run the benchmark to compare backends:
  ```bash
  python scripts/benchmark_webcam_open.py --source 1 --runs 3
  ```
- **"Cannot open video source"**: try a different index (`--source 0`, `--source 1`, `--source 2`) or use a video file path.
- **Black / frozen frames**: launch with `--native` to skip forced resolution/fps settings.

## Technical documentation

The full methodology, experimental results, failure analysis and ethical considerations are in:

- `docs/technical_analysis.md` (source)
- `docs/technical_analysis.pdf` (deliverable)
