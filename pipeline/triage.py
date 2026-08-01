import fitz
import logging

logger = logging.getLogger(__name__)

VECTOR_PATH_THRESHOLD = 50


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
    logger.info(f"Triage page {page_no}: {vector_count} meaningful vector paths, {image_count} embedded images")

    if vector_count >= VECTOR_PATH_THRESHOLD and image_count == 0:
        path, confidence, reason = "vector", min(1.0, vector_count / 500), f"{vector_count} vector paths, no images"
    elif vector_count >= VECTOR_PATH_THRESHOLD and image_count > 0:
        path, confidence, reason = "hybrid", 0.85, f"{vector_count} vector paths + {image_count} images"
    elif vector_count < VECTOR_PATH_THRESHOLD and image_count > 0:
        path, confidence, reason = "raster", min(1.0, image_count / 3), f"Only {vector_count} vector paths, {image_count} images"
    else:
        path, confidence, reason = "raster", 0.5, f"Only {vector_count} vector paths, no images — likely notes page"

    return {"path": path, "vector_path_count": vector_count, "image_count": image_count,
            "has_text": has_text, "confidence": confidence, "reason": reason}


def triage_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    return [triage_page(pdf_path, page_no=i) for i in range(len(doc))]
