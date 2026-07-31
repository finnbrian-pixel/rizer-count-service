"""Stage 0 — PDF Triage: determine if pages are vector, raster, or hybrid."""

import logging
from typing import List, Literal, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

PageRoute = Literal["vector", "raster", "hybrid"]


def triage_page(page: fitz.Page) -> PageRoute:
    """
    Determine whether a page is vector-based, raster-based, or hybrid.

    Checks for vector drawing objects (paths/curves) and image objects.
    """
    drawings = page.get_drawings()
    images = page.get_images(full=True)

    has_vectors = len(drawings) > 10  # Meaningful vector content threshold
    has_images = len(images) > 0

    # Check if images cover significant page area
    significant_raster = False
    if has_images:
        page_area = page.rect.width * page.rect.height
        total_image_area = 0.0
        for img in images:
            xref = img[0]
            try:
                img_rects = page.get_image_rects(xref)
                for rect in img_rects:
                    total_image_area += rect.width * rect.height
            except Exception:
                continue
        if total_image_area > 0.3 * page_area:
            significant_raster = True

    if has_vectors and significant_raster:
        return "hybrid"
    elif has_vectors:
        return "vector"
    else:
        return "raster"


def triage_pdf(doc: fitz.Document) -> List[Tuple[int, PageRoute]]:
    """
    Triage all pages in a PDF document.

    Returns:
        List of (page_index, route) tuples.
    """
    results: List[Tuple[int, PageRoute]] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        route = triage_page(page)
        logger.info(f"Page {page_idx}: routed to '{route}' "
                    f"(drawings={len(page.get_drawings())}, "
                    f"images={len(page.get_images(full=True))})")
        results.append((page_idx, route))
    return results
