# Real-Time Traffic Monitoring

Real-time traffic monitoring system that detects and counts 7 vehicle/pedestrian categories using a custom YOLOv8n model trained on VisDrone2019-DET. Includes a live webcam/video pipeline with a virtual counting line.

## What it does

- **Detection**: locates pedestrians, bicycles, cars, vans, trucks, buses and motorcycles in real time
- **Tracking**: keeps object IDs across frames using an IoU + centroid distance tracker
- **Counting**: counts vehicles crossing a configurable virtual line in either direction

## Tech stack

- Python 3.10+
- OpenCV (video capture and display)
- Ultralytics YOLOv8 (object detection)
- PyTorch + TorchVision (deep learning backend)
- pytest (test suite)

## Requirements

- Python 3.10 or newer
- Git
- Webcam (for live mode) or a video file
- ~500 MB free space (environment + models)
- (Optional but recommended) NVIDIA GPU with CUDA for smoother inference

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

> **Note**: the very first time you run the monitoring pipeline, the YOLO model may take **20-30 seconds** to load. This is normal and depends on your hardware.

### Webcam (built-in — index 0)

```bash
python scripts/run_monitoring.py --source 0
```

### Webcam (external / USB — index 1)

```bash
python scripts/run_monitoring.py --source 1
```

### Video file

```bash
python scripts/run_monitoring.py --source path/to/video.mp4
```

### Demo video (included)

A sample aerial traffic video is included in the repository for quick testing:

```bash
python scripts/run_monitoring.py --source demo/traffic_demo.mp4
```

### Controls while running

- `s` → save screenshot to `docs/results/screenshot.jpg`
- `q` → quit

### CLI options

| Option     | Description                                        | Default |
| ---------- | -------------------------------------------------- | ------- |
| `--source` | Camera index (`0`, `1`...) or path to a video file | `0`     |
| `--width`  | Capture width in pixels                            | `1280`  |
| `--height` | Capture height in pixels                           | `720`   |
| `--native` | Use webcam native resolution (skip width/height)   | `False` |

Example with custom resolution:

```bash
python scripts/run_monitoring.py --source 1 --width 1920 --height 1080
```

## Model weights

Both model files are **included in the repository** and work out of the box:

- **`models/best.pt`** — custom traffic-specific weights trained on VisDrone2019-DET (7 traffic classes). This is the primary model loaded by the application.
- **`models/yolov8n.pt`** — COCO pre-trained weights (fallback). Loaded automatically only if `best.pt` is missing.

No manual download is required to run the demo.

## Configuration

All tunable parameters are centralized in `src/config.py`:

**Counting line** (`WEBCAM` dict):

```python
WEBCAM = {
    "count_line_orientation": "horizontal",  # "horizontal" or "vertical"
    "count_line_y_ratio": 0.5,               # line position: 0.0 (top) to 1.0 (bottom)
    ...
}
```

Change `count_line_orientation` to `"vertical"` for left-to-right counting, and adjust `count_line_y_ratio` to move the line up or down.

**Detection thresholds** (`YOLO` dict):

```python
YOLO = {
    "conf_threshold": 0.5,   # minimum confidence to keep a detection
    "nms_threshold": 0.4,    # NMS IoU threshold
    ...
}
```

**Visualization colors** (`VISUALIZATION` dict):

```python
VISUALIZATION = {
    "box_colors": {
        0: (128, 128, 128),   # pedestrian - gray
        1: (255, 128, 0),     # bicycle - orange
        2: (0, 255, 0),       # car - green
        ...
    },
    "count_line_color": (0, 165, 255),  # orange line
    ...
}
```

## Available scripts

| Script                             | Purpose                                             |
| ---------------------------------- | --------------------------------------------------- |
| `scripts/run_monitoring.py`        | Live webcam / video demo                            |
| `scripts/benchmark_webcam_open.py` | Benchmark webcam open speed with different backends |
| `scripts/download_visdrone.py`     | Download the VisDrone2019-DET dataset               |
| `scripts/build_yolo_dataset.py`    | Convert VisDrone annotations to YOLO format         |
| `scripts/train_yolo.py`            | Train the YOLOv8n traffic detector                  |

## Technical documentation

The full technical analysis (required deliverable) is available in the repository:

- **`docs/Technical_Analysis_Traffic_Monitoring.pdf`** — contains the five required sections: Problem Statement, Methodology, Experimental Results, Failure Analysis, and Ethical Considerations.

## Full pipeline (from scratch)

If you want to re-train the model or start without the included weights, follow these steps.

### 1. Download the VisDrone2019-DET dataset

```bash
python scripts/download_visdrone.py --output data/raw/visdrone
```

Downloads ~1.5 GB of images and annotations.

### 2. Build the structured dataset

```bash
python scripts/build_yolo_dataset.py
```

Converts VisDrone annotations to YOLO format under `data/processed/yolo/`.

### 3. Train the model

```bash
python scripts/train_yolo.py --epochs 25
```

Outputs are saved under `runs/detect/`. Copy the resulting `best.pt` to `models/best.pt` if you want it loaded automatically.

### 4. Launch the demo

```bash
python scripts/run_monitoring.py --source 0
```

## Repository structure

```
├── data/
│   ├── raw/               # VisDrone dataset (gitignored)
│   └── processed/         # YOLO dataset (gitignored)
├── demo/
│   └── traffic_demo.mp4   # Sample aerial traffic video for testing
├── docs/
│   ├── Technical_Analysis_Traffic_Monitoring.pdf  # Required deliverable (PDF)
│   └── results/                                   # Screenshots
├── models/
│   ├── best.pt            # Trained traffic weights (included in repo)
│   └── yolov8n.pt         # COCO fallback (included in repo)
├── scripts/               # Entrypoints for training, building, demo, benchmarks
├── src/
│   ├── config.py          # Centralized configuration
│   ├── deep_learning/     # YOLO detector & training wrapper
│   ├── postprocessing/    # Tracker & counting line logic
│   ├── preprocessing/     # VisDrone → YOLO format conversion
│   ├── utils/             # Drawing utilities (bboxes, counters, line)
│   └── webcam/            # Video streamer and live pipeline
├── tests/                 # pytest suite
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

- **First run is slow**: YOLO weights are loaded at startup. Expect 20-30 seconds before the video window appears.
- **No webcam data leaves the PC**: everything is processed locally.
- **Aerial perspective**: the model is trained on aerial/drone images (VisDrone). Performance on street-level videos may vary.

### Windows-specific webcam issues

- **Webcam index 1 (USB) is very slow to open**: this is a known OpenCV behaviour on Windows with the default DirectShow backend. The project already applies the `cv2.CAP_DSHOW` fix automatically in `src/webcam/streamer.py`. If you still experience slowness, run the benchmark to compare backends:
  ```bash
  python scripts/benchmark_webcam_open.py --source 1 --runs 3
  ```
- **"Cannot open video source"**: try a different index (`--source 0`, `--source 1`, `--source 2`) or use a video file path.
- **Black / frozen frames**: launch with `--native` to skip forced resolution/fps settings.
