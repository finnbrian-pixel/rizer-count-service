import fitz
import logging

logger = logging.getLogger(__name__)

VECTOR_PATH_THRESHOLD = 50

# If vector paths dominate this heavily, incidental images (logos, title block)
# do not make the page "hybrid" — it is a vector drawing.
VECTOR_DOMINANT_THRESHOLD = 500
MAX_INCIDENTAL_IMAGES = 5


def triage_page(pdf_path: str, page_no: int = 0) -> dict:
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    drawings = page.get_drawings()
    meaningful_paths = [
        d for d in drawings
        if d.get("rect") and d["rect"].width > 5 and d["rect"].height > 5
    ]
    images = page.get_images(full=False)
    text = page.get_text().strip()
    has_text = len(text) > 10
    vector_count = len(meaningful_paths)
    image_count = len(images)
    logger.info(
        f"Triage page {page_no}: {vector_count} meaningful vector paths, "
        f"{image_count} embedded images"
    )

    if vector_count >= VECTOR_PATH_THRESHOLD and image_count == 0:
        path = "vector"
        confidence = min(1.0, vector_count / 500)
        reason = f"{vector_count} vector paths, no images"
    elif (vector_count >= VECTOR_DOMINANT_THRESHOLD
          and image_count <= MAX_INCIDENTAL_IMAGES):
        # Overwhelmingly vector content with a few incidental images (logos,
        # title block). Treat as vector — rasterizing would OOM for no benefit.
        path = "vector"
        confidence = min(1.0, vector_count / 500)
        reason = (f"{vector_count} vector paths, only {image_count} incidental "
                  f"images — vector dominant, skipping raster")
    elif vector_count >= VECTOR_PATH_THRESHOLD and image_count > 0:
        path = "hybrid"
        confidence = 0.85
        reason = f"{vector_count} vector paths + {image_count} images"
    elif vector_count < VECTOR_PATH_THRESHOLD and image_count > 0:
        path = "raster"
        confidence = min(1.0, image_count / 3)
        reason = (f"Only {vector_count} vector paths, {image_count} images")
    else:
        path = "raster"
        confidence = 0.5
        reason = (f"Only {vector_count} vector paths, no images — "
                  f"likely notes page")

    result = {
        "path": path,
        "vector_path_count": vector_count,
        "image_count": image_count,
        "has_text": has_text,
        "confidence": confidence,
        "reason": reason,
    }
    logger.info(f"Triage page {page_no} result: path={path}, reason={reason}")
    return result


def triage_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    results = [triage_page(pdf_path, page_no=i) for i in range(len(doc))]
    logger.info(
        f"Triage complete: {len(results)} pages — "
        f"routes: {[r['path'] for r in results]}"
    )
    return results
