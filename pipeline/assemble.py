"""Stage 4 — Count Assembly: combine detection results into final counts."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def estimate_legend_bbox(
    page_width: float, page_height: float
) -> Tuple[float, float, float, float]:
    """
    Estimate the legend/title block region (typically bottom-right of sheet).
    Returns (x0, y0, x1, y1) in page coordinates.
    """
    x0 = page_width * 0.75
    y0 = page_height * 0.70
    x1 = page_width
    y1 = page_height
    return (x0, y0, x1, y1)


def is_in_exclusion_zone(
    pos: Tuple[float, float],
    legend_bbox_norm: Tuple[float, float, float, float],
) -> bool:
    """Check if a normalized position is within the legend/title block exclusion zone."""
    x, y = pos[0], pos[1]
    x0, y0, x1, y1 = legend_bbox_norm
    return x0 <= x <= x1 and y0 <= y <= y1


def compute_confidence_from_scores(scores: List[float]) -> float:
    """
    Derive confidence from match score statistics.
    Higher mean and lower variance = higher confidence.
    """
    if not scores:
        return 0.0
    mean_score = np.mean(scores)
    std_score = np.std(scores) if len(scores) > 1 else 0.0
    confidence = mean_score * (1.0 - min(std_score * 2, 0.3))
    return round(float(np.clip(confidence, 0.0, 1.0)), 4)


def assemble_counts(
    vector_positions: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    raster_positions: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    classification: Optional[Dict[str, Any]] = None,
    page_width: float = 1.0,
    page_height: float = 1.0,
    sheet_name: str = "SHEET-1",
    path_used: str = "vector",
    corrections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble final count results from detection positions and classifications.

    Args:
        vector_positions: Dict of cluster_id -> list of (norm_x, norm_y).
        raster_positions: Dict of cluster_id -> list of (norm_x, norm_y, score).
        classification: Claude classification result.
        page_width: Page width in points.
        page_height: Page height in points.
        sheet_name: Sheet identifier.
        path_used: "vector", "raster", or "hybrid".
        corrections: Optional user corrections from prior verification.

    Returns:
        Stage 4 count assembly result dict.
    """
    # Build classification lookup
    class_lookup: Dict[str, Dict[str, Any]] = {}
    if classification:
        for entry in classification.get("cluster_classification", []):
            cid = entry.get("cluster_id", "")
            if cid:
                class_lookup[cid] = entry

    # Apply corrections if provided
    if corrections:
        for cid, correction in corrections.items():
            if cid in class_lookup:
                class_lookup[cid].update(correction)

    # Estimate legend exclusion zone (normalized)
    legend_bbox_norm = (0.75, 0.70, 1.0, 1.0)

    # Process positions
    counts_by_type: Dict[str, Dict[str, Any]] = {}
    flags: List[str] = []
    all_overlay_positions: List[Dict[str, Any]] = []
    all_scores: List[float] = []

    # Process vector positions
    if vector_positions:
        for cluster_id, positions in vector_positions.items():
            cls_info = class_lookup.get(cluster_id, {})
            classification_type = cls_info.get("classification", "sprinkler_head")
            head_type = cls_info.get("head_type", "unknown")

            if classification_type != "sprinkler_head":
                continue

            valid_positions: List[List[float]] = []
            excluded_count = 0
            for pos in positions:
                if is_in_exclusion_zone(pos, legend_bbox_norm):
                    excluded_count += 1
                    continue
                valid_positions.append([pos[0], pos[1]])
                all_overlay_positions.append({
                    "x": pos[0],
                    "y": pos[1],
                    "cluster_id": cluster_id,
                    "head_type": head_type,
                    "confidence": 1.0,
                    "source": "vector",
                })

            if excluded_count > 0:
                flags.append(
                    f"cluster {cluster_id} near legend excluded "
                    f"({excluded_count} inside legend bbox)"
                )

            if head_type not in counts_by_type:
                counts_by_type[head_type] = {"count": 0, "positions": []}
            counts_by_type[head_type]["count"] += len(valid_positions)
            counts_by_type[head_type]["positions"].extend(valid_positions)

    # Process raster positions
    if raster_positions:
        for cluster_id, detections in raster_positions.items():
            cls_info = class_lookup.get(cluster_id, {})
            classification_type = cls_info.get("classification", "sprinkler_head")
            head_type = cls_info.get("head_type", "unknown")

            if classification_type != "sprinkler_head":
                continue

            valid_positions: List[List[float]] = []
            low_confidence_count = 0
            for det in detections:
                norm_x, norm_y = det[0], det[1]
                score = det[2] if len(det) > 2 else 0.9

                if is_in_exclusion_zone((norm_x, norm_y), legend_bbox_norm):
                    continue

                valid_positions.append([norm_x, norm_y])
                conf = float(score)
                all_scores.append(conf)

                is_low_conf = score < 0.85
                if is_low_conf:
                    low_confidence_count += 1

                all_overlay_positions.append({
                    "x": norm_x,
                    "y": norm_y,
                    "cluster_id": cluster_id,
                    "head_type": head_type,
                    "confidence": round(conf, 4),
                    "source": "raster",
                    "low_confidence": is_low_conf,
                })

            if low_confidence_count > 0:
                flags.append(
                    f"{low_confidence_count} template matches below 0.85 score "
                    f"-- highlight for verification"
                )

            if head_type not in counts_by_type:
                counts_by_type[head_type] = {"count": 0, "positions": []}
            counts_by_type[head_type]["count"] += len(valid_positions)
            counts_by_type[head_type]["positions"].extend(valid_positions)

    # Build counts list
    counts_list = [
        {"head_type": ht, "count": data["count"], "positions": data["positions"]}
        for ht, data in counts_by_type.items()
    ]

    # Compute overall confidence
    if path_used == "vector":
        confidence = 1.0
    elif all_scores:
        confidence = compute_confidence_from_scores(all_scores)
    else:
        confidence = 0.0

    # Check hybrid discrepancy
    needs_verification = False
    if path_used == "hybrid" and vector_positions and raster_positions:
        vector_total = sum(len(p) for p in vector_positions.values())
        raster_total = sum(len(d) for d in raster_positions.values())
        if vector_total > 0:
            discrepancy = abs(vector_total - raster_total) / vector_total
            if discrepancy > 0.02:
                needs_verification = True
                flags.append(
                    f"Hybrid discrepancy: vector={vector_total}, raster={raster_total} "
                    f"({discrepancy:.1%} difference)"
                )

    # Total count
    total = sum(item["count"] for item in counts_list)

    # Silent zero is a critical failure mode
    if total == 0:
        return {
            "sheet": sheet_name,
            "total_heads": 0,
            "confidence": 0.0,
            "needs_verification": True,
            "flags": flags + [
                "Zero heads detected -- verify this is not a counting failure"
            ],
            "counts": counts_list,
            "path_used": path_used,
            "overlay": {
                "positions": [],
                "page_width": page_width,
                "page_height": page_height,
                "color_coding": {
                    "high_confidence": "green",
                    "low_confidence": "amber",
                    "threshold": 0.85,
                },
                "low_confidence_positions": [],
            },
        }

    # Confidence guardrail
    if confidence < 0.90:
        needs_verification = True
        if not any("confidence" in f.lower() for f in flags):
            flags.append(
                f"Low confidence ({confidence:.2f} < 0.90) -- needs human verification"
            )

    result: Dict[str, Any] = {
        "sheet": sheet_name,
        "counts": counts_list,
        "total_heads": total,
        "flags": flags,
        "confidence": confidence,
        "path_used": path_used,
        "needs_verification": needs_verification,
    }

    # Stage 5 overlay data
    result["overlay"] = {
        "positions": all_overlay_positions,
        "page_width": page_width,
        "page_height": page_height,
        "color_coding": {
            "high_confidence": "green",
            "low_confidence": "amber",
            "threshold": 0.85,
        },
        "low_confidence_positions": [
            p for p in all_overlay_positions if p.get("low_confidence", False)
        ],
    }

    return result
