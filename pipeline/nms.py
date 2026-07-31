"""Non-max suppression utility for template matching results."""

import numpy as np
from typing import List, Tuple


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two boxes (x, y, w, h)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(
    detections: List[Tuple[int, int, float]],
    template_w: int,
    template_h: int,
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """
    Apply non-max suppression to a list of detections.

    Args:
        detections: List of (x, y, score) tuples.
        template_w: Width of the matched template.
        template_h: Height of the matched template.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Filtered list of (x, y, score) tuples after NMS.
    """
    if not detections:
        return []

    # Sort by score descending
    sorted_dets = sorted(detections, key=lambda d: d[2], reverse=True)
    kept: List[Tuple[int, int, float]] = []

    for det in sorted_dets:
        x, y, score = det
        box = (x, y, template_w, template_h)
        suppressed = False

        for kept_det in kept:
            kx, ky, _ = kept_det
            kept_box = (kx, ky, template_w, template_h)
            if compute_iou(box, kept_box) > iou_threshold:
                suppressed = True
                break

        if not suppressed:
            kept.append(det)

    return kept
