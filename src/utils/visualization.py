"""Drawing utilities for bounding boxes, labels, counters."""

import cv2
import numpy as np
from src.config import VISUALIZATION, TRAFFIC_CLASSES


def draw_detections(image, detections, class_names=None):
    """Draw boxes and labels."""
    if class_names is None:
        class_names = TRAFFIC_CLASSES

    img_h, img_w = image.shape[:2]
    out = image.copy()
    box_colors = VISUALIZATION.get("box_colors", {})
    default_color = (0, 255, 0)

    for det in detections:
        class_id, conf, x, y, w, h = det
        class_id = int(class_id)

        if max(x, y, w, h) <= 1.0:
            x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        label = f"{class_names[class_id]}: {conf:.2f}"
        color = box_colors.get(class_id, default_color)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, VISUALIZATION["thickness"])

        (tw, th), _ = cv2.getTextSize(
            label, VISUALIZATION["font"], VISUALIZATION["font_scale"], 1
        )
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)

        cv2.putText(
            out,
            label,
            (x1, y1 - 2),
            VISUALIZATION["font"],
            VISUALIZATION["font_scale"],
            VISUALIZATION["text_color"],
            1,
            cv2.LINE_AA,
        )

    return out


def draw_count_line(image, orientation="horizontal", ratio=0.5, color=None, thickness=None):
    """Draw counting line."""
    out = image.copy()
    h, w = image.shape[:2]
    color = color or VISUALIZATION.get("count_line_color", (0, 165, 255))
    thickness = thickness or VISUALIZATION.get("line_thickness", 2)

    if orientation == "horizontal":
        y = int(h * ratio)
        cv2.line(out, (0, y), (w, y), color, thickness)
        cv2.putText(
            out,
            "COUNT LINE",
            (10, y - 10),
            VISUALIZATION["font"],
            0.6,
            color,
            thickness,
            cv2.LINE_AA,
        )
    else:
        x = int(w * ratio)
        cv2.line(out, (x, 0), (x, h), color, thickness)
        cv2.putText(
            out,
            "COUNT LINE",
            (x + 10, 30),
            VISUALIZATION["font"],
            0.6,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return out


def draw_counters(image, counts, class_names=None):
    """Draw count stats on top-right."""
    out = image.copy()
    if class_names is None:
        class_names = TRAFFIC_CLASSES

    h, w = image.shape[:2]
    box_colors = VISUALIZATION.get("box_colors", {})
    default_color = (0, 255, 0)

    # Normalize counts keys to class names
    named_counts = {}
    for k, v in counts.items():
        if isinstance(k, int) and 0 <= k < len(class_names):
            named_counts[class_names[k]] = v
        else:
            named_counts[str(k)] = v

    total = sum(named_counts.values())
    lines = ["--- COUNTS ---", f"TOTAL: {total}"] + [
        f"{name}: {count}" for name, count in named_counts.items()
    ]

    # Compute max width for background box
    max_tw = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, VISUALIZATION["font"], 0.6, 1)
        max_tw = max(max_tw, tw)

    line_height = 22
    box_h = len(lines) * line_height + 10
    box_w = max_tw + 20
    x0 = w - box_w - 10
    y0 = 10

    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        y = y0 + 25 + i * line_height
        if i >= 2:
            class_id = i - 2
            color = box_colors.get(class_id, default_color)
        else:
            color = (255, 255, 255)
        cv2.putText(out, line, (x0 + 10, y), VISUALIZATION["font"], 0.6, color, 1, cv2.LINE_AA)

    return out
