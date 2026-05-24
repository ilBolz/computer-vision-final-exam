"""
Simple object tracker for traffic counting.

Implements a centroid/IOU-based tracker that maintains object IDs
across frames and triggers counting events when objects cross a
virtual counting line.
"""

from collections import defaultdict

import numpy as np


class SimpleTracker:
    """IOU-based tracker with counting logic."""

    def __init__(self, iou_threshold=0.3, max_missing=5):
        """
        Initialize tracker.

        Args:
            iou_threshold: Minimum IoU to match a detection to an existing track.
            max_missing: Maximum frames a track can be missing before deletion.
        """
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_id = 0
        self.tracks = {}  # id -> {centroid, bbox, missing_count, counted, class_id}

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: List of [class_id, confidence, x, y, w, h].

        Returns:
            List of active tracks, each as dict with keys:
                id, class_id, bbox, centroid, counted.
        """
        new_centroids = []
        new_bboxes = []
        for det in detections:
            cls_id, conf, x, y, w, h = det
            cx = x + w / 2.0
            cy = y + h / 2.0
            new_centroids.append((cx, cy))
            new_bboxes.append((x, y, w, h))

        matched = set()
        used_tracks = set()

        # Greedy matching by IoU
        for i, bbox in enumerate(new_bboxes):
            best_iou = 0.0
            best_tid = None
            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._compute_iou(bbox, track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["centroid"] = new_centroids[i]
                self.tracks[best_tid]["missing_count"] = 0
                self.tracks[best_tid]["class_id"] = int(detections[i][0])
                used_tracks.add(best_tid)
                matched.add(i)

        # Unmatched detections -> new tracks
        for i in range(len(new_bboxes)):
            if i not in matched:
                self.tracks[self.next_id] = {
                    "bbox": new_bboxes[i],
                    "centroid": new_centroids[i],
                    "missing_count": 0,
                    "counted": False,
                    "class_id": int(detections[i][0]),
                }
                self.next_id += 1

        # Unmatched tracks -> increment missing
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["missing_count"] += 1
                if self.tracks[tid]["missing_count"] > self.max_missing:
                    del self.tracks[tid]

        return [
            {
                "id": tid,
                "class_id": t["class_id"],
                "bbox": t["bbox"],
                "centroid": t["centroid"],
                "counted": t["counted"],
            }
            for tid, t in self.tracks.items()
        ]

    def check_crossings(self, tracks, line_orientation="horizontal", line_ratio=0.5, img_h=720, img_w=1280):
        """
        Check which tracks have crossed the counting line and update counters.

        Args:
            tracks: Output from update().
            line_orientation: 'horizontal' or 'vertical'.
            line_ratio: Position of line as ratio (0-1).
            img_h: Image height.
            img_w: Image width.

        Returns:
            counts: Dict {class_id: increment} for this frame.
        """
        counts = defaultdict(int)

        if line_orientation == "horizontal":
            line_pos = img_h * line_ratio
            for track in tracks:
                if track["counted"]:
                    continue
                _, cy = track["centroid"]
                # Trigger when centroid crosses line from top to bottom
                # Simple logic: if previously above and now below
                # For simplicity in this exam project, we count once when
                # the centroid passes the line (state stored in track).
                if cy >= line_pos:
                    track["counted"] = True
                    counts[track["class_id"]] += 1
        else:
            line_pos = img_w * line_ratio
            for track in tracks:
                if track["counted"]:
                    continue
                cx, _ = track["centroid"]
                if cx >= line_pos:
                    track["counted"] = True
                    counts[track["class_id"]] += 1

        # Update internal track state for counted flag
        for track in tracks:
            tid = track["id"]
            if tid in self.tracks:
                self.tracks[tid]["counted"] = track["counted"]

        return dict(counts)

    @staticmethod
    def _compute_iou(box_a, box_b):
        """Compute IoU between two xywh boxes (center-based)."""
        x1, y1, w1, h1 = box_a
        x2, y2, w2, h2 = box_b

        # Convert to xyxy
        ax1, ay1, ax2, ay2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        bx1, by1, bx2, by2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = w1 * h1
        area_b = w2 * h2
        union = area_a + area_b - inter_area
        return inter_area / (union + 1e-6)
