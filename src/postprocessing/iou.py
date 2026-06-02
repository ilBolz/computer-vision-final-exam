"""IoU computation for bounding boxes."""

import numpy as np


def convert_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(box_a, box_b, format="xywh") -> float:
    """Compute IoU between two boxes. Supports xywh and xyxy."""
    if format == "xywh":
        box_a = convert_xywh_to_xyxy(box_a)
        box_b = convert_xywh_to_xyxy(box_b)

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def compute_iou_matrix(boxes_a: list, boxes_b: list, format="xywh") -> np.ndarray:
    """Pairwise IoU matrix between two lists of boxes."""
    n = len(boxes_a)
    m = len(boxes_b)
    matrix = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            matrix[i, j] = compute_iou(boxes_a[i], boxes_b[j], format=format)
    return matrix
