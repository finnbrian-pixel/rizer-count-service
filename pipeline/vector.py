"""Stage 1A — Vector Extraction: extract drawing primitives and fingerprint them."""

import hashlib
import io
import logging
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def normalize_path_commands(path: Dict[str, Any]) -> str:
    """
    Normalize a path by translating to origin and rounding coordinates to 1 decimal place.
    Returns a canonical string representation suitable for hashing.
    """
    items = path.get("items", [])
    if not items:
        return ""

    # Find the minimum x, y to translate to origin
    all_points: List[Tuple[float, float]] = []
    for item in items:
        # Each item is a tuple like ("l", Point, Point) or ("c", P1, P2, P3, P4) etc.
        for element in item[1:]:
            if hasattr(element, "x") and hasattr(element, "y"):
                all_points.append((element.x, element.y))
            elif isinstance(element, (tuple, list)) and len(element) >= 2:
                all_points.append((float(element[0]), float(element[1])))

    if not all_points:
        return ""

    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)

    # Rebuild normalized representation
    normalized_parts: List[str] = []
    for item in items:
        cmd = item[0]
        coords: List[str] = []
        for element in item[1:]:
            if hasattr(element, "x") and hasattr(element, "y"):
                nx = round(element.x - min_x, 1)
                ny = round(element.y - min_y, 1)
                coords.append(f"{nx},{ny}")
            elif isinstance(element, (tuple, list)) and len(element) >= 2:
                nx = round(float(element[0]) - min_x, 1)
                ny = round(float(element[1]) - min_y, 1)
                coords.append(f"{nx},{ny}")
        normalized_parts.append(f"{cmd}:{';'.join(coords)}")

    return "|".join(normalized_parts)


def compute_fingerprint(path: Dict[str, Any]) -> str:
    """Compute a SHA256 fingerprint of the normalized path commands."""
    normalized = normalize_path_commands(path)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def is_closed_path(path: Dict[str, Any]) -> bool:
    """Determine if a path is closed (forms a shape)."""
    items = path.get("items", [])
    if not items:
        return False
    # Check if path has a closePath command or first/last points match
    if path.get("closePath", False):
        return True
    # Check 'type' field from PyMuPDF drawings
    if path.get("type", "") == "f":  # filled = closed
        return True
    # If items start and end at same point
    if len(items) >= 2:
        first_item = items[0]
        last_item = items[-1]
        if len(first_item) >= 2 and len(last_item) >= 2:
            start = first_item[1] if len(first_item) > 1 else None
            end = last_item[-1] if len(last_item) > 1 else None
            if start and end and hasattr(start, "x") and hasattr(end, "x"):
                if abs(start.x - end.x) < 0.5 and abs(start.y - end.y) < 0.5:
                    return True
    return False


def get_path_bbox(path: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Get bounding box of a path as (x0, y0, x1, y1)."""
    rect = path.get("rect", None)
    if rect:
        if hasattr(rect, "x0"):
            return (rect.x0, rect.y0, rect.x1, rect.y1)
        return tuple(rect[:4])
    # Fallback: compute from items
    all_x: List[float] = []
    all_y: List[float] = []
    for item in path.get("items", []):
        for element in item[1:]:
            if hasattr(element, "x"):
                all_x.append(element.x)
                all_y.append(element.y)
    if all_x and all_y:
        return (min(all_x), min(all_y), max(all_x), max(all_y))
    return (0, 0, 0, 0)


def extract_vector_clusters(
    page: fitz.Page, min_cluster_size: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract drawing primitives, fingerprint them, and cluster by fingerprint.

    Args:
        page: A PyMuPDF page object.
        min_cluster_size: Minimum instances to keep a cluster (filters noise).

    Returns:
        Dict mapping fingerprint -> list of path instances with bounding boxes.
    """
    drawings = page.get_drawings()
    logger.info(f"Found {len(drawings)} drawing primitives on page")

    # Fingerprint all paths
    fingerprint_map: Dict[str, List[Dict[str, Any]]] = {}

    for drawing in drawings:
        # Filter very small or very large paths (likely borders/frames)
        bbox = get_path_bbox(drawing)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Skip paths that are too small (< 3pt) or too large (> 100pt)
        if width < 3 or height < 3:
            continue
        if width > 100 or height > 100:
            continue

        fp = compute_fingerprint(drawing)
        if not fp:
            continue

        entry = {
            "path": drawing,
            "bbox": bbox,
            "fingerprint": fp,
        }

        if fp not in fingerprint_map:
            fingerprint_map[fp] = []
        fingerprint_map[fp].append(entry)

    # Filter to clusters with min_cluster_size instances
    clusters = {
        fp: instances
        for fp, instances in fingerprint_map.items()
        if len(instances) >= min_cluster_size
    }

    logger.info(
        f"Clustered into {len(fingerprint_map)} unique fingerprints, "
        f"{len(clusters)} clusters with >= {min_cluster_size} instances"
    )

    return clusters


def render_cluster_sample(
    page: fitz.Page, bbox: Tuple[float, float, float, float], padding: float = 5.0
) -> bytes:
    """
    Render a small region around a path instance to a PNG image.

    Args:
        page: The PyMuPDF page.
        bbox: Bounding box (x0, y0, x1, y1) of the path instance.
        padding: Extra padding around the bbox.

    Returns:
        PNG image bytes of the rendered region.
    """
    x0, y0, x1, y1 = bbox
    clip_rect = fitz.Rect(x0 - padding, y0 - padding, x1 + padding, y1 + padding)

    # Render at 4x zoom for clarity
    mat = fitz.Matrix(4, 4)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    return pix.tobytes("png")


def get_cluster_positions(
    clusters: Dict[str, List[Dict[str, Any]]],
    page_width: float,
    page_height: float,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Get normalized (0-1) positions for all instances in each cluster.

    Returns:
        Dict mapping fingerprint -> list of (norm_x, norm_y) positions.
    """
    positions: Dict[str, List[Tuple[float, float]]] = {}
    for fp, instances in clusters.items():
        fp_positions: List[Tuple[float, float]] = []
        for inst in instances:
            bbox = inst["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            norm_x = cx / page_width if page_width > 0 else 0
            norm_y = cy / page_height if page_height > 0 else 0
            fp_positions.append((round(norm_x, 6), round(norm_y, 6)))
        positions[fp] = fp_positions
    return positions
