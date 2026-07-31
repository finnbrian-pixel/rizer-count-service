"""Stage 1B — Template Matching: raster-based sprinkler head detection."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

from .nms import non_max_suppression

logger = logging.getLogger(__name__)


def rasterize_page(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """
    Rasterize a PDF page to a grayscale numpy array.

    Args:
        page: PyMuPDF page object.
        dpi: Resolution for rasterization.

    Returns:
        Grayscale image as numpy array.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return img


def preprocess_image(gray: np.ndarray) -> np.ndarray:
    """
    Apply adaptive threshold and morphological open to normalize scan noise.
    """
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # Morphological open to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees."""
    if angle == 0:
        return image.copy()
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def match_heads(
    sheet_gray: np.ndarray,
    template_gray: np.ndarray,
    threshold: float = 0.80,
) -> List[Tuple[int, int, float]]:
    """
    Multi-scale, multi-rotation template matching.

    Args:
        sheet_gray: Grayscale image of the full sheet.
        template_gray: Grayscale image of the template symbol.
        threshold: Minimum match score to accept.

    Returns:
        List of (x, y, score) tuples after NMS.
    """
    results: List[Tuple[int, int, float]] = []
    th, tw = template_gray.shape[:2]

    for scale in np.linspace(0.85, 1.15, 7):
        new_w = max(int(tw * scale), 1)
        new_h = max(int(th * scale), 1)
        t = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        for angle in (0, 90, 180, 270):
            tr = rotate_image(t, angle)
            tr_h, tr_w = tr.shape[:2]

            # Skip if template is larger than sheet
            if tr_h >= sheet_gray.shape[0] or tr_w >= sheet_gray.shape[1]:
                continue

            res = cv2.matchTemplate(sheet_gray, tr, cv2.TM_CCOEFF_NORMED)
            locations = np.where(res >= threshold)
            ys, xs = locations

            for x, y in zip(xs, ys):
                results.append((int(x), int(y), float(res[y, x])))

    # Apply NMS
    if not results:
        return []

    # Use average template size for NMS box calculation
    avg_w = int(tw * 1.0)
    avg_h = int(th * 1.0)
    filtered = non_max_suppression(results, avg_w, avg_h, iou_threshold=0.3)

    logger.info(f"Template matching: {len(results)} raw -> {len(filtered)} after NMS")
    return filtered


def run_template_matching(
    page: fitz.Page,
    templates: Dict[str, np.ndarray],
    threshold: float = 0.80,
    dpi: int = 200,
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Run template matching for all templates on a rasterized page.

    Args:
        page: PyMuPDF page object.
        templates: Dict of cluster_id -> template grayscale image.
        threshold: Match threshold.
        dpi: Rasterization DPI.

    Returns:
        Dict of cluster_id -> list of (x, y, score) matches.
    """
    sheet_gray = rasterize_page(page, dpi=dpi)
    sheet_processed = preprocess_image(sheet_gray)

    results: Dict[str, List[Tuple[int, int, float]]] = {}
    score_histogram: Dict[str, List[float]] = {}

    for cluster_id, template in templates.items():
        # Also preprocess template
        if template.ndim == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        template_processed = preprocess_image(template)
        matches = match_heads(sheet_processed, template_processed, threshold=threshold)
        results[cluster_id] = matches

        # Log score histogram
        scores = [m[2] for m in matches]
        score_histogram[cluster_id] = scores
        if scores:
            logger.info(
                f"Cluster {cluster_id}: {len(scores)} matches, "
                f"score range [{min(scores):.3f}, {max(scores):.3f}], "
                f"mean={np.mean(scores):.3f}"
            )

    return results


def get_raster_positions_normalized(
    matches: Dict[str, List[Tuple[int, int, float]]],
    page_width_px: int,
    page_height_px: int,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Convert pixel positions to normalized 0-1 coordinates.

    Returns:
        Dict of cluster_id -> list of (norm_x, norm_y, score).
    """
    normalized: Dict[str, List[Tuple[float, float, float]]] = {}
    for cluster_id, detections in matches.items():
        norm_dets: List[Tuple[float, float, float]] = []
        for x, y, score in detections:
            norm_x = x / page_width_px if page_width_px > 0 else 0
            norm_y = y / page_height_px if page_height_px > 0 else 0
            norm_dets.append((round(norm_x, 6), round(norm_y, 6), round(score, 4)))
        normalized[cluster_id] = norm_dets
    return normalized
