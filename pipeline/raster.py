import logging
from collections import Counter

import fitz

logger = logging.getLogger(__name__)

DPI = 300  # hard floor — never lower

MAX_PIXELS = 40_000_000  # 40 megapixels — hard ceiling

# Blank-paper threshold: a real symbol has mean ~218 (dark ink on white).
# Anything above 235 is blank paper and must be rejected.
BLANK_THRESHOLD = 235


def _apply_pixel_budget(page, dpi: int) -> int:
    """
    Given a PyMuPDF page and requested DPI, return the effective DPI
    that keeps total pixel count under MAX_PIXELS.
    """
    pw_pts = page.rect.width
    ph_pts = page.rect.height

    # Pixels at requested DPI
    px_w = pw_pts * dpi / 72.0
    px_h = ph_pts * dpi / 72.0
    total_px = px_w * px_h

    # If over budget, scale down DPI to fit
    effective_dpi = dpi
    if total_px > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / total_px) ** 0.5
        effective_dpi = max(72, int(dpi * scale_factor))
        logger.warning(
            f"Page would render to {int(total_px/1e6):.0f}M px at {dpi} DPI. "
            f"Downscaling to {effective_dpi} DPI "
            f"({int(px_w*scale_factor):.0f}x{int(px_h*scale_factor):.0f} px)"
        )

    return effective_dpi


def rasterize(pdf_path: str, page_no: int = 0, dpi: int = DPI):
    import cv2
    import numpy as np
    """Rasterize a PDF page to grayscale numpy array at `dpi`.
    
    Returns:
        (gray, effective_dpi) — the grayscale image and the actual DPI used
        (may be lower than requested if the pixel budget cap was applied).
    """
    doc = fitz.open(pdf_path)
    page = doc[page_no]

    # Apply pixel budget cap
    effective_dpi = _apply_pixel_budget(page, dpi)

    pix = page.get_pixmap(dpi=effective_dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img.copy()

    # Always log final render dimensions and effective DPI
    logger.info(
        f"Rasterized page {page_no}: {gray.shape[1]}x{gray.shape[0]} px "
        f"at effective {effective_dpi} DPI (requested {dpi} DPI, "
        f"budget {MAX_PIXELS/1e6:.0f}M px)"
    )
    return gray, effective_dpi


def rasterize_page(page: fitz.Page, dpi: int = DPI):
    """Rasterize a PyMuPDF page object to grayscale numpy array.
    
    Returns:
        (gray, effective_dpi) — the grayscale image and the actual DPI used
        (may be lower than requested if the pixel budget cap was applied).
    """
    # Apply pixel budget cap
    effective_dpi = _apply_pixel_budget(page, dpi)

    pix = page.get_pixmap(dpi=effective_dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img.copy()

    logger.info(
        f"Rasterized page: {gray.shape[1]}x{gray.shape[0]} px "
        f"at effective {effective_dpi} DPI (requested {dpi} DPI, "
        f"budget {MAX_PIXELS/1e6:.0f}M px)"
    )
    return gray, effective_dpi


def _fingerprint_blob(crop: np.ndarray) -> str:
    """
    Produce a size-invariant fingerprint for a small grayscale crop.
    Resizes to 16x16, binarizes, returns hex hash of the bit pattern.
    """
    resized = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 200, 1, cv2.THRESH_BINARY_INV)
    # Pack 256 bits into a hex string
    bits = binary.flatten().astype(np.uint8)
    byte_arr = np.packbits(bits)
    return byte_arr.tobytes().hex()


def find_template_by_clustering(
    sheet_gray: np.ndarray,
    min_blob: int = 20,
    max_blob: int = 80,
    min_cluster_size: int = 10,
) -> np.ndarray:
    """
    Scan the entire sheet for the most-repeated compact symbol shape.
    This is the raster equivalent of Path A vector clustering:
    1. Binary threshold to find all ink blobs
    2. Filter to symbol-sized blobs (min_blob..max_blob px in each dim)
    3. Fingerprint each blob (size-invariant hash)
    4. Take the most frequent fingerprint cluster
    5. Return the first instance of that cluster as the template

    No legend location needed — the sprinkler head is by definition the
    most-repeated small shape on a fire protection plan.
    """
    logger.info(
        f"find_template_by_clustering: scanning sheet {sheet_gray.shape} "
        f"for repeated blobs [{min_blob}-{max_blob}px]"
    )

    # Binary threshold
    _, binary = cv2.threshold(sheet_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Collect symbol-sized blobs
    blobs = []  # (fingerprint, crop, label_idx)
    for i in range(1, num_labels):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if not (min_blob <= bw <= max_blob and min_blob <= bh <= max_blob):
            continue
        # Reject very elongated shapes (text characters, lines)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 2.5:
            continue

        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        pad = 3
        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(sheet_gray.shape[1], bx + bw + pad)
        y1 = min(sheet_gray.shape[0], by + bh + pad)
        crop = sheet_gray[y0:y1, x0:x1]

        fp = _fingerprint_blob(crop)
        blobs.append((fp, crop, i))

    if not blobs:
        raise ValueError(
            "find_template_by_clustering: no symbol-sized blobs found on sheet"
        )

    # Count fingerprints
    fp_counts = Counter(fp for fp, _, _ in blobs)
    most_common_fp, count = fp_counts.most_common(1)[0]

    logger.info(
        f"  Found {len(blobs)} candidate blobs, "
        f"{len(fp_counts)} unique fingerprints. "
        f"Top cluster: count={count}, min_required={min_cluster_size}"
    )

    if count < min_cluster_size:
        raise ValueError(
            f"find_template_by_clustering: largest cluster has only {count} instances "
            f"(need >= {min_cluster_size}). Sheet may not contain repeated symbols."
        )

    # Get the first instance of the dominant cluster as template
    for fp, crop, _ in blobs:
        if fp == most_common_fp:
            mean_val = float(crop.mean())
            if mean_val > BLANK_THRESHOLD:
                continue  # skip if somehow blank
            logger.info(
                f"  Template selected from dominant cluster: "
                f"shape={crop.shape}, mean={mean_val:.1f}, cluster_size={count}"
            )
            return crop

    raise ValueError(
        "find_template_by_clustering: dominant cluster instances all blank"
    )


def crop_template(sheet_gray: np.ndarray, legend_bbox_pts, dpi: int = DPI) -> np.ndarray:
    """
    Crop a template from sheet_gray using a legend bbox in PDF point coordinates.
    PDF points must be scaled to raster pixels: scale = dpi / 72.0

    If the primary bbox lands on blank paper (mean > 235), falls back to
    full-sheet clustering: finds the most-repeated compact shape on the
    entire page and uses that as the template.
    """
    scale = dpi / 72.0

    x0 = int(legend_bbox_pts[0] * scale)
    y0 = int(legend_bbox_pts[1] * scale)
    x1 = int(legend_bbox_pts[2] * scale)
    y1 = int(legend_bbox_pts[3] * scale)

    # Diagnostic log: bbox in both coordinate systems
    logger.info(
        f"Legend bbox: PDF pts={list(legend_bbox_pts)}, "
        f"raster px=[{x0},{y0},{x1},{y1}] scale={scale:.3f}"
    )

    # Clamp to image bounds
    h, w = sheet_gray.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)

    if x1 <= x0 or y1 <= y0:
        logger.warning(
            f"Legend bbox produced empty region after clamping. "
            f"Falling back to full-sheet clustering."
        )
        return find_template_by_clustering(sheet_gray)

    tpl = sheet_gray[y0:y1, x0:x1]

    if tpl is None or tpl.size == 0:
        logger.warning("Legend crop is zero-size. Falling back to full-sheet clustering.")
        return find_template_by_clustering(sheet_gray)

    mean_val = float(tpl.mean())
    if mean_val > BLANK_THRESHOLD:
        logger.warning(
            f"Legend crop is blank (mean={mean_val:.1f} > {BLANK_THRESHOLD}). "
            f"Coordinate-space bug likely still present. "
            f"Falling back to full-sheet clustering."
        )
        return find_template_by_clustering(sheet_gray)

    logger.info(f"Template acquired (primary legend bbox): shape={tpl.shape}, mean={mean_val:.1f}")
    return tpl


def nms(dets: list, iou_thresh: float = 0.30) -> list:
    import numpy as np
    """
    Non-maximum suppression.
    dets: list of (x, y, w, h, score)
    Returns filtered list.
    """
    if not dets:
        return []
    boxes = np.array(
        [[d[0], d[1], d[0] + d[2], d[1] + d[3]] for d in dets], dtype=float
    )
    scores = np.array([d[4] for d in dets])
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_o = (
            (boxes[idx[1:], 2] - boxes[idx[1:], 0])
            * (boxes[idx[1:], 3] - boxes[idx[1:], 1])
        )
        idx = idx[1:][inter / (a_i + a_o - inter) < iou_thresh]
    return [dets[i] for i in keep]


def match_heads(
    sheet_gray: np.ndarray, tpl_gray: np.ndarray, threshold: float = 0.80
) -> list:
    """
    Multi-scale, multi-rotation template matching with NMS.
    Verified working params: 300 DPI, 48x48 template, TM_CCOEFF_NORMED, thresh=0.80, NMS IoU=0.30
    Scale sweep: 0.85->1.15 (7 steps). Rotation: 0/90/180/270.
    """
    # Guardrail: assert template validity
    if tpl_gray is None or tpl_gray.size == 0:
        raise ValueError("empty template -- legend detection failed upstream")
    if tpl_gray.mean() > BLANK_THRESHOLD:
        raise ValueError(
            f"blank template (mean={tpl_gray.mean():.1f} > {BLANK_THRESHOLD}) "
            f"-- crop landed on empty paper"
        )

    dets = []
    scores_all = []

    for scale in np.linspace(0.85, 1.15, 7):
        t = cv2.resize(tpl_gray, None, fx=scale, fy=scale)
        if t.shape[0] > sheet_gray.shape[0] or t.shape[1] > sheet_gray.shape[1]:
            continue
        for k in range(4):  # 0, 90, 180, 270 degrees
            tr = np.rot90(t, k).copy()
            res = cv2.matchTemplate(sheet_gray, tr, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= threshold)
            for x, y in zip(xs, ys):
                score = float(res[y, x])
                dets.append((int(x), int(y), tr.shape[1], tr.shape[0], score))
                scores_all.append(score)

    # Log score histogram for diagnostics (bimodal = high confidence)
    if scores_all:
        arr = np.array(scores_all)
        logger.info(
            f"Score histogram: min={arr.min():.2f} mean={arr.mean():.2f} "
            f"max={arr.max():.2f} n={len(arr)} raw_dets={len(dets)}"
        )

    result = nms(dets)
    logger.info(f"After NMS: {len(result)} detections")
    return result


def run_template_matching(
    page: fitz.Page,
    templates: dict,
    dpi: int = DPI,
    threshold: float = 0.80,
) -> dict:
    """
    Run template matching on a rasterized page with given templates.

    Args:
        page: PyMuPDF page object.
        templates: Dict of cluster_id -> grayscale template (numpy array).
        dpi: DPI for rasterization.
        threshold: Match threshold.

    Returns:
        Dict of cluster_id -> list of (x, y, w, h, score) detections.
    """
    sheet_gray, effective_dpi = rasterize_page(page, dpi=dpi)
    results = {}

    for cluster_id, tpl in templates.items():
        if tpl is None or tpl.size == 0:
            continue
        if tpl.mean() > BLANK_THRESHOLD:
            logger.warning(
                f"Skipping blank template for cluster {cluster_id} "
                f"(mean={tpl.mean():.1f})"
            )
            continue
        try:
            hits = match_heads(sheet_gray, tpl, threshold=threshold)
            results[cluster_id] = hits
            logger.info(f"Cluster {cluster_id}: {len(hits)} matches")
        except ValueError as e:
            logger.warning(f"Template matching failed for {cluster_id}: {e}")
            results[cluster_id] = []

    return results


def get_raster_positions_normalized(
    matches: dict,
    px_width: int,
    px_height: int,
) -> dict:
    """
    Convert raster match positions to normalized (0-1) coordinates.

    Args:
        matches: Dict of cluster_id -> list of (x, y, w, h, score) detections.
        px_width: Sheet width in pixels.
        px_height: Sheet height in pixels.

    Returns:
        Dict of cluster_id -> list of (norm_x, norm_y, score) tuples.
    """
    normalized = {}
    for cluster_id, dets in matches.items():
        positions = []
        for det in dets:
            x, y, w, h, score = det
            cx = (x + w / 2.0) / px_width
            cy = (y + h / 2.0) / px_height
            positions.append((round(cx, 6), round(cy, 6), round(score, 4)))
        normalized[cluster_id] = positions
    return normalized


def sanity_check(detections: int, sheet_gray: np.ndarray) -> dict:
    """
    Guardrail 3: sanity floor.
    If detections are suspiciously low relative to sheet area, flag for review.
    """
    h, w = sheet_gray.shape
    sheet_area_px2 = h * w
    expected_floor = sheet_area_px2 / 50000
    needs_review = detections < max(1, expected_floor * 0.01)

    return {
        "detections": detections,
        "expected_floor": expected_floor,
        "needs_review": needs_review,
        "flag": (
            "Suspiciously low detection count -- verify template and legend detection"
            if needs_review
            else None
        ),
    }
