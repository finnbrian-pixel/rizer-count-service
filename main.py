"""
Rizer Count Service — FastAPI microservice for deterministic sprinkler head counting.

AI (Claude) is used ONLY for legend classification. Counting is purely algorithmic
via vector fingerprinting or template matching.
"""

import io
import logging
import os
import shutil
import tempfile
import traceback
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.assemble import assemble_counts
from pipeline.triage import triage_pdf
from pipeline.vector import count_heads, count_in_region

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rizer Count Service",
    description="Deterministic sprinkler head counting pipeline for fire protection blueprints.",
    version="1.0.0",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/count")
async def count_sprinkler_heads(
    pdf: UploadFile = File(..., description="PDF blueprint file"),
    corrections: Optional[str] = Form(
        None, description="JSON string of prior user corrections"
    ),
):
    """
    Count sprinkler heads in a PDF blueprint.

    Pipeline:
    1. Triage pages (vector vs raster)
    2. Extract/detect symbols via count_heads (vector) or template matching (raster)
    3. Assemble deterministic counts
    4. Return overlay data for verification UI

    Accepts optional `corrections` JSON from prior user verification.
    """
    # Validate file
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail="File must be a PDF. Received: " + (pdf.filename or "no filename"),
        )

    # Parse corrections if provided
    corrections_dict: Optional[Dict[str, Any]] = None
    if corrections:
        try:
            import json
            corrections_dict = json.loads(corrections)
        except (json.JSONDecodeError, TypeError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid corrections JSON: {e}",
            )

    # Stream upload directly to temp file — never buffer full bytes in memory
    tmp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            shutil.copyfileobj(pdf.file, tmp_pdf)
            tmp_pdf_path = tmp_pdf.name
            tmp_pdf_size = tmp_pdf.tell()
    except Exception as e:
        if tmp_pdf_path:
            try:
                os.unlink(tmp_pdf_path)
            except Exception:
                pass
        raise HTTPException(status_code=422, detail=f"Failed to read PDF: {e}")

    if tmp_pdf_size == 0:
        os.unlink(tmp_pdf_path)
        raise HTTPException(status_code=422, detail="Empty PDF file")

    # Open PDF with PyMuPDF from disk path — no in-memory stream copy
    try:
        doc = fitz.open(tmp_pdf_path)
    except Exception as e:
        os.unlink(tmp_pdf_path)
        raise HTTPException(
            status_code=422,
            detail=f"Corrupt or invalid PDF: {e}",
        )

    if len(doc) == 0:
        doc.close()
        os.unlink(tmp_pdf_path)
        raise HTTPException(status_code=422, detail="PDF has no pages")

    try:
        # Stage 0 — Triage
        logger.info(f"Processing PDF: {pdf.filename}, {len(doc)} pages")

        page_routes = triage_pdf(tmp_pdf_path)
        logger.info(
            f"Triage results: {[(i, r['path'], r['reason']) for i, r in enumerate(page_routes)]}"
        )

        # Collect page dimensions upfront, then close the doc to free memory
        # before count_heads opens its own handle on the same file.
        page_dims = [(doc[i].rect.width, doc[i].rect.height) for i in range(len(doc))]
        doc.close()

        results: list = []

        for page_idx, triage_result in enumerate(page_routes):
            route = triage_result["path"]
            page_width, page_height = page_dims[page_idx]
            sheet_name = f"SHEET-{page_idx + 1}"

            logger.info(
                f"Processing {sheet_name} via '{route}' path "
                f"(page={page_width:.0f}x{page_height:.0f} pt)"
            )

            vector_count_result = None

            # Stage 1A — Vector Counting
            if route in ("vector", "hybrid"):
                try:
                    vector_count_result = count_heads(tmp_pdf_path, page_no=page_idx)
                    logger.info(
                        f"{sheet_name}: count_heads -> {vector_count_result['total']} heads "
                        f"(confidence={vector_count_result['confidence']:.2f})"
                    )
                except Exception as e:
                    logger.error(f"{sheet_name}: count_heads failed: {e}")

            # Stage 1B — Raster path
            if route in ("raster", "hybrid") and vector_count_result is None:
                # Pure raster path: no vector result available.
                # Template matching without cluster samples is not supported;
                # fall through to assemble_counts with no positions so the
                # caller receives needs_verification=True and can prompt the user.
                logger.warning(
                    f"{sheet_name}: Raster path active but no vector result available. "
                    "Skipping template matching — needs_verification will be set."
                )

            # Stage 2 — Build page result
            if vector_count_result is not None:
                # Build the page result directly from count_heads output.
                # This covers vector and hybrid routes.
                page_result = {
                    "sheet_name": sheet_name,
                    "total_heads": vector_count_result["total"],
                    "counts": vector_count_result["counts"],
                    "heads": vector_count_result["heads"],
                    "rejected": vector_count_result["rejected"],
                    "confidence": vector_count_result["confidence"],
                    "needs_verification": vector_count_result["needs_verification"],
                    "path_used": route,
                    "method": vector_count_result.get("method", "vector-geometric"),
                }
            else:
                # Raster-only fallback: assemble with no positions
                page_result = assemble_counts(
                    vector_positions=None,
                    raster_positions=None,
                    classification=None,
                    page_width=page_width,
                    page_height=page_height,
                    sheet_name=sheet_name,
                    path_used=route,
                    corrections=corrections_dict,
                )

            results.append(page_result)

        # Aggregate results
        total_heads = sum(r["total_heads"] for r in results)
        overall_confidence = (
            min(r["confidence"] for r in results) if results else 0.0
        )
        any_needs_verification = any(
            r.get("needs_verification", False) for r in results
        )

        # Collect routing info for caller visibility
        paths_used = [r.get("path_used", "unknown") for r in results]

        response = {
            "filename": pdf.filename,
            "pages_processed": len(results),
            "total_heads": total_heads,
            "confidence": overall_confidence,
            "needs_verification": any_needs_verification,
            "path_used": paths_used[0] if len(set(paths_used)) == 1 else "mixed",
            "sheets": results,
        }

        logger.info(
            f"Completed: {pdf.filename} — {total_heads} heads across "
            f"{len(results)} pages, confidence={overall_confidence:.2f}"
        )

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except MemoryError:
        logger.error("OOM: PDF raster resolution exceeds memory limits")
        return JSONResponse(
            status_code=422,
            content={
                "error": "File too large to process",
                "reason": "PDF raster resolution exceeds memory limits even after downscaling",
                "needs_verification": True,
            },
        )
    except Exception as e:
        logger.exception("Pipeline failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Processing failed",
                "reason": str(e),
                "needs_verification": True,
            },
        )
    finally:
        # doc already closed above after collecting page dims;
        # guard in case of early exception before that point.
        try:
            doc.close()
        except Exception:
            pass
        # Clean up temp file
        try:
            os.unlink(tmp_pdf_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
