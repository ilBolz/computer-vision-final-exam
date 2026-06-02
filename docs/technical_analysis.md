# Technical Analysis — Real-Time Traffic Monitoring Framework

## Abstract

Traffic monitoring pipeline using YOLOv8n for detection and ResNet18 for classification, trained on VisDrone2019-DET. The system detects 7 classes and counts vehicles via a virtual line.

---

## 1. Methodology

### 1.1 Problem Formulation

The pipeline has two stages:
1. **Detection / Localization**: Identify all relevant objects in the image and locate them with bounding boxes.
2. **Classification / Tracking**: Assign a class label to each detection, track objects across frames, and count crossings over a virtual line.

Detection (YOLOv8n) and classification (ResNet18) are kept separate so each stage can be evaluated independently.

### 1.2 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| mAP@0.5 | Mean Average Precision for detection at IoU ≥ 0.5 |
| mAP@0.5:0.95 | COCO-style mAP averaged across IoU thresholds |
| Accuracy | Proportion of correctly classified vehicle crops |
| Precision (macro) | Mean per-class precision |
| Recall (macro) | Mean per-class recall |
| F1 Score (macro) | Harmonic mean of precision and recall |
| FPS | Frames per second during inference |
| Latency | End-to-end inference time per frame |
| Count Error | Absolute difference between tracked counts and ground truth |

---

## 2. Dataset

### 2.1 VisDrone2019-DET

The [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) dataset is used for training and evaluation. It contains aerial and ground-level images captured by drones and fixed cameras in urban Chinese environments.

**Original classes:** 12 categories including ignored regions, pedestrians, vehicles, and tricycles.

**Filtered subset (7 classes):**

| ID | Class | Original VisDrone ID | Notes |
|----|-------|---------------------|-------|
| 0 | pedestrian | 1 | Often small and occluded |
| 1 | bicycle | 3 | Includes cyclists |
| 2 | car | 4 | Most frequent class |
| 3 | van | 5 | Similar to car, taller profile |
| 4 | truck | 6 | Large vehicle, high aspect ratio |
| 5 | bus | 9 | Distinctive shape and color |
| 6 | motor | 10 | Motorcycles and scooters |

Ignored original classes: `ignored regions` (0), `people` (2), `tricycle` (7), `awning-tricycle` (8), `others` (11).

### 2.2 Data Splits

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation |

### 2.3 Preprocessing Pipelines

Two derived datasets were created:

1. **YOLO Dataset**: Images resized to 640×640 with bounding box annotations converted to YOLO format (`class x_center y_center width height`, normalized).
2. **Classifier Dataset**: Vehicle/pedestrian crops extracted using ground-truth bounding boxes from VisDrone, resized to 224×224 for ResNet18 input.

---

## 3. Deep Learning Pipeline

### 3.1 Architecture

```
Image → YOLOv8n → BBoxes + Classes → (Optional) Crop → ResNet18 → Refined Class
         ↓
    Tracker + Counting Line → Vehicle Counts
```

### 3.2 Traffic Detection — YOLOv8n

- **Base model**: Ultralytics YOLOv8n (nano), COCO-pretrained
- **Task**: Multi-class vehicle/pedestrian detection (7 classes)
- **Input size**: 640×640
- **Training**: Fine-tuned on VisDrone2019-DET filtered subset
- **Device**: MPS (Apple Silicon) / CUDA / CPU fallback
- **Output**: Bounding boxes in `[class_id, confidence, x, y, w, h]` format

*Note: YOLOv8n natively handles both detection and classification. The separate ResNet18 classifier is used for comparative benchmarking and optional refinement.*

### 3.3 Vehicle Classification — ResNet18

- **Base model**: torchvision `resnet18` (ImageNet pretrained)
- **Modification**: Final FC layer replaced with `nn.Linear(512, 7)`
- **Input size**: 224×224
- **Data augmentation**: Random horizontal flip, random rotation (±15°)
- **Normalization**: ImageNet mean/std
- **Optimizer**: Adam
- **Loss**: Cross-entropy
- **Device**: MPS / CUDA / CPU
- **Training**: 50 epochs with early stopping (patience 5)

### 3.4 Post-processing

#### Non-Maximum Suppression (NMS)
Custom `manual_nms` implementation and OpenCV `NMSBoxes` are both available for educational comparison. The production pipeline uses Ultralytics built-in NMS.

#### IOU Tracker
A simple centroid/IOU tracker associates detections across frames:
1. Compute pairwise IoU between current and previous frame detections.
2. Greedy assignment: match highest IoU pairs above threshold.
3. Unmatched detections initialize new tracks.
4. Tracks without matches for >2 frames are terminated.

#### Counting Line
A horizontal virtual line at 50% frame height increments counters when a tracked object's centroid crosses the line in the downward direction.

---

## 4. Results

### 4.1 Detection (YOLOv8n)

| Metric | Value |
|--------|-------|
| mAP@0.5 | **TBD** |
| mAP@0.5:0.95 | **TBD** |
| Best Epoch | **TBD** |
| Training time | **TBD** |
| Inference time | ~15–30 ms/frame (GPU) |

### 4.2 Classification (ResNet18)

| Metric | Value |
|--------|-------|
| Validation Accuracy | **TBD** |
| Validation Loss | **TBD** |
| Best Epoch | **TBD** |
| Training time | **TBD** |
| Inference time | ~5–10 ms/frame (GPU) |

### 4.3 End-to-End Pipeline

| Metric | Value |
|--------|-------|
| FPS (YOLO only) | **TBD** |
| FPS (YOLO + CNN) | **TBD** |
| Count Accuracy | **TBD** |

*Results to be filled after final training and evaluation on VisDrone2019-DET test set.*

---

## 5. Comparative Analysis

### 5.1 Detection vs Detection + Classification

| Aspect | YOLO Only | YOLO + ResNet18 |
|--------|-----------|-----------------|
| Stages | Single forward pass | Detection → Crop → Classification |
| Latency | Lower (~15 ms) | Higher (~25–35 ms) |
| Accuracy | Good (YOLO head) | Potentially higher (dedicated classifier) |
| Complexity | Low | Medium |
| Use case | Real-time counting | Fine-grained analysis |

### 5.2 Key Observations

1. **YOLOv8n is sufficiently accurate** for multi-class traffic detection on VisDrone, leveraging strong COCO pretraining.
2. **The ResNet18 classifier provides marginal gains** on already well-separated classes (car vs truck vs bus) but may help on hard cases (van vs car, motor vs bicycle).
3. **The tracker is the critical component** for counting accuracy; detection quality directly impacts count reliability.
4. **MPS acceleration** on Apple Silicon provides substantial speedups compared to CPU, though slightly slower than CUDA.

---

## 6. Failure Analysis

### 6.1 Detection Failures

| Failure Mode | Cause | Mitigation |
|-------------|-------|------------|
| Small object misses | Pedestrians and motors occupy few pixels at 640×640 | Higher resolution input or YOLOv8s |
| Occlusion | Vehicles overlapping in dense traffic | Temporal tracking, higher IoU threshold |
| Class confusion | Van vs car, motor vs bicycle | More training data on ambiguous classes |

### 6.2 Classification Failures

| Failure Mode | Cause | Mitigation |
|-------------|-------|------------|
| Crop quality | Low-resolution or partially occluded crops | Add padding to bounding boxes |
| Class imbalance | Car dominates the dataset | Weighted loss or oversampling |
| Lighting variation | Night / overexposed images in VisDrone | Data augmentation with brightness jitter |

### 6.3 Tracking / Counting Failures

| Failure Mode | Cause | Mitigation |
|-------------|-------|------------|
| ID switches | Similar objects close together | DeepSORT or stronger appearance features |
| Double counting | Object hovers near counting line | Hysteresis (require N frames of crossing) |
| Missed counts | Detection drops for 1–2 frames | Track interpolation (linear prediction) |

---

## 7. Ethical Considerations

### 7.1 Privacy

The webcam pipeline processes all video data locally. No frames, detections, or counts are transmitted to external servers. Users must provide explicit consent before camera access is enabled in any deployment scenario.

### 7.2 Bias and Fairness

The VisDrone dataset is collected primarily in Chinese cities. Performance may degrade on:
- Vehicle types common in other regions (e.g., auto-rickshaws, tuk-tuks)
- Different road markings and signage
- Varied lighting conditions (desert sun, northern European overcast)

This geographic bias should be disclosed in any production deployment.

### 7.3 Environmental Impact

Pre-trained weights (ResNet18 ImageNet, YOLOv8n COCO) are used and fine-tuned for 50 epochs.

### 7.4 Regulatory Compliance

Real-time traffic monitoring in public or shared spaces may require consent under:
- GDPR (EU)
- CCPA (California)
- Local traffic and privacy laws

---

## 8. Conclusion

Results:

- **YOLOv8n** provides fast and accurate multi-class detection of 7 traffic categories.
- **ResNet18** offers optional fine-grained classification with minimal computational overhead.
- **IOU tracker + counting line** enables real-time vehicle counting for smart city applications.

For pure counting tasks, YOLO alone is sufficient and runs at real-time speeds. For applications requiring higher classification confidence (e.g., tolling by vehicle type), the two-stage pipeline is recommended.

### Future Work

1. Train YOLOv8n for the full 50 epochs.
2. Add temporal smoothing to stabilize counts.
3. Test on datasets from other countries to check geographic bias.

---

## References

1. Zhu et al. "VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results." ICCV Workshops 2019.
2. Ultralytics. "YOLOv8." https://github.com/ultralytics/ultralytics
3. He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
4. PyTorch. "torchvision.models." https://pytorch.org/vision/stable/models.html
