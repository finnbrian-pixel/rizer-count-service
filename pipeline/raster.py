import logging
import numpy as np
import cv2
import fitz

logger = logging.getLogger(__name__)

DPI = 300  # hard floor — never lower

def rasterize(pdf_path: str, page_no: int = 0, dpi: int = DPI):
    """Rasterize a PDF page to grayscale numpy array at `dpi`."""
    doc = fitz.open(pdf_path)
    pix = doc[page_no].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img.copy()
    logger.info(f"Rasterized page {page_no} at {dpi} DPI \u2192 {gray.shape}")
    return gray


def crop_template(sheet_gray: np.ndarray, legend_bbox_pts, dpi: int = DPI) -> np.ndarray:
    """
    Crop a template from sheet_gray using a legend bbox in PDF point coordinates.
    PDF points must be scaled to raster pixels: scale = dpi / 72.0
    """
    scale = dpi / 72.0
    x0 = int(legend_bbox_pts[0] * scale)
    y0 = int(legend_bbox_pts[1] * scale)
    x1 = int(legend_bbox_pts[2] * scale)
    y1 = int(legend_bbox_pts[3] * scale)

    # Clamp to image bounds
    h, w = sheet_gray.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)

    tpl = sheet_gray[y0:y1, x0:x1]

    # Guardrail: blank template = coordinate space bug or empty legend region
    if tpl is None or tpl.size == 0:
        raise ValueError("Template crop is empty \u2014 check legend bbox coordinates")
    if tpl.mean() > 250:
        raise ValueError(
            f"Template crop is blank (mean={tpl.mean():.1f}) \u2014 "
            f"legend bbox coordinate space mismatch or empty legend region. "
            f"bbox_pts={legend_bbox_pts}, scale={scale:.2f}, "
            f"cropped px=[{x0}:{x1}, {y0}:{y1}]"
        )

    logger.info(f"Template acquired: shape={tpl.shape}, mean={tpl.mean():.1f}")
    return tpl


def nms(dets: list, iou_thresh: float = 0.30) -> list:
    """
    Non-maximum suppression.
    dets: list of (x, y, w, h, score)
    Returns filtered list.
    """
    if not dets:
        return []
    boxes = np.array([[d[0], d[1], d[0] + d[2], d[1] + d[3]] for d in dets], dtype=float)
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
        a_o = (boxes[idx[1:], 2] - boxes[idx[1:], 0]) * (boxes[idx[1:], 3] - boxes[idx[1:], 1])
        idx = idx[1:][inter / (a_i + a_o - inter) < iou_thresh]
    return [dets[i] for i in keep]


def match_heads(sheet_gray: np.ndarray, tpl_gray: np.ndarray, threshold: float = 0.80) -> list:
    """
    Multi-scale, multi-rotation template matching with NMS.
    Verified working params: 300 DPI, 48x48 template, TM_CCOEFF_NORMED, thresh=0.80, NMS IoU=0.30
    Scale sweep: 0.85\u21921.15 (7 steps). Rotation: 0/90/180/270.
    """
    # Guardrail: assert template validity
    if tpl_gray is None or tpl_gray.size == 0:
        raise ValueError("empty template \u2014 legend detection failed upstream")
    if tpl_gray.mean() > 250:
        raise ValueError(f"blank template (mean={tpl_gray.mean():.1f}) \u2014 crop landed on empty paper")

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


def sanity_check(detections: int, sheet_gray: np.ndarray) -> dict:
    """
    Guardrail 3: sanity floor.
    If detections are suspiciously low relative to sheet area, flag for review.
    Heuristic: a typical FP sheet at 1/8 scale has ~1 head per 150 sq ft.
    At 300 DPI, 1 sq ft \u2248 (300/12)^2 = 625 px\u00b2. Use loose threshold.
    """
    h, w = sheet_gray.shape
    sheet_area_px2 = h * w
    # Very rough: expect at least 1 head per 50,000 px\u00b2 on a real FP plan
    expected_floor = sheet_area_px2 / 50000
    needs_review = detections < max(1, expected_floor * 0.01)  # less than 1% of heuristic floor

    return {
        "detections": detections,
        "expected_floor": expected_floor,
        "needs_review": needs_review,
        "flag": "Suspiciously low detection count \u2014 verify template and legend detection" if needs_review else None
    }
