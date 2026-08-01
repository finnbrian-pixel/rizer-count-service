"""
TEST-FP-103 Scan Simulation
Creates a synthetic fire protection plan PDF with:
- 175 sprinkler head symbols spread across the plan area
- A legend in the bottom-right with 2 example symbols
- Tests that the pipeline correctly:
  1. Finds a valid template (mean < 235) via clustering fallback
  2. Detects 175 hits via template matching
  3. Excludes 2 legend symbols -> final count = 173
"""
import logging
import math
import sys

import cv2
import fitz
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TEST-FP-103")

sys.path.insert(0, "/root/workspace/rizer-count-service")
from pipeline.raster import (
    BLANK_THRESHOLD,
    crop_template,
    find_template_by_clustering,
    match_heads,
    rasterize,
)
from pipeline.assemble import assemble_counts, is_in_exclusion_zone

# ---- Test-harness constants (NOT in production code) ----
EXPECTED_TOTAL_HEADS = 173
EXPECTED_TOLERANCE = 2


def create_test_pdf(output_path: str, n_heads: int = 175, n_legend: int = 2):
    """Create a synthetic FP plan: n_heads symbols total, n_legend in the legend zone."""
    page_w_pts = 2448
    page_h_pts = 1584

    doc = fitz.open()
    page = doc.new_page(width=page_w_pts, height=page_h_pts)

    # Border
    page.draw_rect(fitz.Rect(10, 10, page_w_pts - 10, page_h_pts - 10), color=(0, 0, 0), width=2)

    # Legend box bottom-right
    legend_x0 = page_w_pts * 0.80
    legend_y0 = page_h_pts * 0.78
    page.draw_rect(fitz.Rect(legend_x0, legend_y0, page_w_pts - 20, page_h_pts - 20), color=(0, 0, 0), width=1)
    page.insert_text(fitz.Point(legend_x0 + 10, legend_y0 + 20), "LEGEND", fontsize=14)

    symbol_radius = 6

    def draw_symbol(pg, cx, cy):
        pg.draw_circle(fitz.Point(cx, cy), symbol_radius, color=(0, 0, 0), width=1.5)
        pg.draw_line(fitz.Point(cx - symbol_radius, cy), fitz.Point(cx + symbol_radius, cy), color=(0, 0, 0), width=1)
        pg.draw_line(fitz.Point(cx, cy - symbol_radius), fitz.Point(cx, cy + symbol_radius), color=(0, 0, 0), width=1)

    # Legend symbols
    legend_sy = legend_y0 + 50
    for i in range(n_legend):
        lx = legend_x0 + 30 + i * 60
        draw_symbol(page, lx, legend_sy)
        page.insert_text(fitz.Point(lx + 12, legend_sy + 4), f"TYPE {i+1}", fontsize=8)

    # Main area symbols
    main_heads = n_heads - n_legend
    main_x_min, main_x_max = 50, page_w_pts * 0.74
    main_y_min, main_y_max = 50, page_h_pts * 0.68

    cols = int(math.ceil(math.sqrt(main_heads * (main_x_max - main_x_min) / (main_y_max - main_y_min))))
    rows = int(math.ceil(main_heads / cols))
    x_step = (main_x_max - main_x_min) / (cols + 1)
    y_step = (main_y_max - main_y_min) / (rows + 1)

    placed = 0
    for row in range(rows):
        for col in range(cols):
            if placed >= main_heads:
                break
            draw_symbol(page, main_x_min + (col + 1) * x_step, main_y_min + (row + 1) * y_step)
            placed += 1
        if placed >= main_heads:
            break

    logger.info(f"Created test PDF: {placed} main + {n_legend} legend = {placed + n_legend} total")
    doc.save(output_path)
    doc.close()
    return page_w_pts, page_h_pts


def run_test():
    pdf_path = "/tmp/TEST-FP-103-scan-sim.pdf"

    logger.info("=" * 60)
    logger.info("TEST-FP-103 Scan Simulation")
    logger.info("=" * 60)

    page_w_pts, page_h_pts = create_test_pdf(pdf_path, n_heads=175, n_legend=2)

    # Rasterize
    sheet_gray = rasterize(pdf_path, page_no=0, dpi=300)
    logger.info(f"Sheet: {sheet_gray.shape}, mean={sheet_gray.mean():.1f}")

    # Test 1: crop_template with intentionally BAD bbox (top-left, blank area)
    # This tests the clustering fallback
    logger.info("\n--- TEST 1: crop_template with bad bbox (triggers clustering fallback) ---")
    bad_bbox = [10, 10, 100, 100]  # top-left corner, blank paper
    tpl = crop_template(sheet_gray, bad_bbox, dpi=300)
    tpl_mean = float(tpl.mean())
    logger.info(f"Template from fallback: shape={tpl.shape}, mean={tpl_mean:.1f}")
    assert tpl_mean < BLANK_THRESHOLD, f"Template mean {tpl_mean} >= {BLANK_THRESHOLD}"
    logger.info("PASS: clustering fallback found valid template")

    # Test 2: find_template_by_clustering directly
    logger.info("\n--- TEST 2: find_template_by_clustering directly ---")
    tpl_cluster = find_template_by_clustering(sheet_gray)
    tpl_cluster_mean = float(tpl_cluster.mean())
    logger.info(f"Clustered template: shape={tpl_cluster.shape}, mean={tpl_cluster_mean:.1f}")
    assert tpl_cluster_mean < BLANK_THRESHOLD

    # Test 3: template matching
    logger.info("\n--- TEST 3: template matching ---")
    hits = match_heads(sheet_gray, tpl_cluster, threshold=0.80)
    hit_count = len(hits)
    logger.info(f"Raw hits: {hit_count}")

    # Test 4: legend exclusion
    logger.info("\n--- TEST 4: legend exclusion ---")
    h_px, w_px = sheet_gray.shape
    legend_bbox_norm = (0.75, 0.70, 1.0, 1.0)

    main_hits = []
    legend_hits = []
    for det in hits:
        x, y, w, h, score = det
        norm_x = (x + w / 2.0) / w_px
        norm_y = (y + h / 2.0) / h_px
        if is_in_exclusion_zone((norm_x, norm_y), legend_bbox_norm):
            legend_hits.append(det)
        else:
            main_hits.append(det)

    logger.info(f"Main area: {len(main_hits)}, Legend zone: {len(legend_hits)}")

    # Test 5: assemble (no expected_heads in production — we assert here in harness)
    logger.info("\n--- TEST 5: assembly ---")
    raster_positions = {
        "head_cluster_0": [
            ((x + w / 2.0) / w_px, (y + h / 2.0) / h_px, score)
            for x, y, w, h, score in hits
        ]
    }
    classification = {
        "cluster_classification": [{
            "cluster_id": "head_cluster_0",
            "classification": "sprinkler_head",
            "head_type": "pendent",
        }],
    }

    result = assemble_counts(
        raster_positions=raster_positions,
        classification=classification,
        page_width=page_w_pts,
        page_height=page_h_pts,
        sheet_name="TEST-FP-103",
        path_used="raster",
    )

    # ---- RESULTS ----
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Template mean:        {tpl_cluster_mean:.1f} (threshold < {BLANK_THRESHOLD})")
    logger.info(f"Raw hits:             {hit_count}")
    logger.info(f"After legend excl:    {result['total_heads']}")
    logger.info(f"Confidence:           {result['confidence']:.4f}")
    logger.info(f"Needs verification:   {result['needs_verification']}")
    logger.info(f"Flags:                {result.get('flags', [])}")

    # ---- TEST HARNESS ASSERTIONS (not in production) ----
    logger.info("\n--- ASSERTIONS (test harness only) ---")
    tpl_ok = tpl_cluster_mean < BLANK_THRESHOLD
    hits_ok = abs(hit_count - 175) <= 5
    final_ok = abs(result['total_heads'] - EXPECTED_TOTAL_HEADS) <= EXPECTED_TOLERANCE
    conf_ok = result['confidence'] >= 0.90
    verif_ok = result['needs_verification'] is False

    logger.info(f"Template mean < {BLANK_THRESHOLD}:  {'PASS' if tpl_ok else 'FAIL'} ({tpl_cluster_mean:.1f})")
    logger.info(f"Raw hits ~175:         {'PASS' if hits_ok else 'FAIL'} ({hit_count})")
    logger.info(f"Final == 173 (+-2):    {'PASS' if final_ok else 'FAIL'} ({result['total_heads']})")
    logger.info(f"Confidence >= 0.90:    {'PASS' if conf_ok else 'FAIL'} ({result['confidence']:.4f})")
    logger.info(f"needs_verification=F:  {'PASS' if verif_ok else 'FAIL'} ({result['needs_verification']})")

    all_pass = tpl_ok and hits_ok and final_ok and conf_ok and verif_ok
    logger.info(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return {
        "template_mean": tpl_cluster_mean,
        "raw_hits": hit_count,
        "final_count": result["total_heads"],
        "confidence": result["confidence"],
        "needs_verification": result["needs_verification"],
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    results = run_test()
    print(
        f"\nFINAL: template_mean={results['template_mean']:.1f}, "
        f"hits={results['raw_hits']}, final={results['final_count']}, "
        f"conf={results['confidence']:.4f}, all_pass={results['all_pass']}"
    )
    sys.exit(0 if results["all_pass"] else 1)
