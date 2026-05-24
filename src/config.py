"""
Centralized configuration for the Real-Time Traffic Monitoring Framework.

Single source of truth for paths, class names, and hyperparameters.
Import this module instead of hard-coding values.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Processed subdirectories
YOLO_DATASET_DIR = PROCESSED_DATA_DIR / "yolo"
CLASSIFIER_DATASET_DIR = PROCESSED_DATA_DIR / "classifier"
KEYPOINTS_DATASET_DIR = PROCESSED_DATA_DIR / "keypoints"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_TRAFFIC_MODEL_PATH = MODELS_DIR / "yolov8n_traffic.pt"
YOLO_TRAFFIC_MODEL_FALLBACK = (
    PROJECT_ROOT / "runs" / "detect" / "models" / "yolov8n_traffic" / "weights" / "best.pt"
)
VEHICLE_CNN_PATH = MODELS_DIR / "vehicle_cnn.pt"
HOG_SVM_PATH = MODELS_DIR / "hog_svm.pkl"

# Traffic classes (VisDrone2019-DET filtered subset)
# Original VisDrone IDs mapped to continuous indices:
# 1 (pedestrian) -> 0
# 3 (bicycle)    -> 1
# 4 (car)        -> 2
# 5 (van)        -> 3
# 6 (truck)      -> 4
# 9 (bus)        -> 5
# 10 (motor)     -> 6
TRAFFIC_CLASSES = ["pedestrian", "bicycle", "car", "van", "truck", "bus", "motor"]
NUM_TRAFFIC_CLASSES = len(TRAFFIC_CLASSES)

# VisDrone original to filtered class mapping
VISDRONE_CLASS_MAP = {
    1: 0,   # pedestrian
    3: 1,   # bicycle
    4: 2,   # car
    5: 3,   # van
    6: 4,   # truck
    9: 5,   # bus
    10: 6,  # motor
}
VISDRONE_IGNORE_CLASSES = {0, 2, 7, 8, 11}  # ignored, people, tricycle, awning-tricycle, others

# Ensure directories exist
for d in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ANNOTATIONS_DIR,
    YOLO_DATASET_DIR,
    CLASSIFIER_DATASET_DIR,
    KEYPOINTS_DATASET_DIR,
    MODELS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Preprocessing parameters
PREPROCESSING = {
    "yolo_input_size": 640,
    "classifier_input_size": 224,
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet
    "normalize_std": [0.229, 0.224, 0.225],   # ImageNet
    "legacy_yolo_input_size": (416, 416),      # Width, Height for legacy YOLO
    "legacy_scalefactor": 1 / 255.0,
    "legacy_mean": (0, 0, 0),
    "legacy_swap_rb": True,
    "legacy_crop": False,
}

# Classical pipeline parameters (HOG + SVM)
CLASSICAL = {
    "hog_win_size": (64, 128),      # Width, Height
    "hog_block_size": (16, 16),
    "hog_block_stride": (8, 8),
    "hog_cell_size": (8, 8),
    "hog_nbins": 9,
    "svm_kernel": "rbf",
    "svm_c": 1.0,
    "scale_factor": 1.05,           # Image pyramid scale for sliding window
    "min_neighbors": 3,             # Min neighbors for grouping (custom NMS-like)
    "stride": (8, 8),               # Sliding window stride
}

# YOLO parameters
YOLO = {
    "model_name": "yolov8n.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.01,
    "conf_threshold": 0.5,
    "nms_threshold": 0.4,
    "single_class": False,  # Multi-class traffic detection
    "num_classes": NUM_TRAFFIC_CLASSES,
}

# CNN parameters
CNN = {
    "architecture": "resnet18",  # or "custom"
    "num_classes": NUM_TRAFFIC_CLASSES,
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001,
    "patience": 5,
}

# Evaluation parameters
EVALUATION = {
    "iou_threshold": 0.5,
    "map_iou_thresholds": [0.5, 0.75],
    "traffic_classes": TRAFFIC_CLASSES,
}

# Visualization parameters
VISUALIZATION = {
    "box_colors": {
        0: (128, 128, 128),   # pedestrian - gray
        1: (255, 128, 0),     # bicycle - orange
        2: (0, 255, 0),       # car - green
        3: (255, 255, 0),     # van - cyan/yellow
        4: (0, 0, 255),       # truck - red
        5: (255, 0, 255),     # bus - magenta
        6: (0, 128, 255),     # motor - light blue
    },
    "text_color": (255, 255, 255),
    "count_line_color": (0, 165, 255),  # orange line
    "font": 0,  # cv2.FONT_HERSHEY_SIMPLEX
    "font_scale": 0.5,
    "thickness": 2,
    "line_thickness": 2,
}

# Webcam parameters
WEBCAM = {
    "source": 0,
    "width": 1280,
    "height": 720,
    "fps": 30,
    "count_line_y_ratio": 0.5,   # Horizontal count line at 50% of frame height
    "count_line_orientation": "horizontal",  # or "vertical"
}
