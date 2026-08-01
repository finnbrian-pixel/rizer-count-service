"""
Rizer Count Service — FastAPI microservice for deterministic sprinkler head counting.

AI (Claude) is used ONLY for legend classification. Counting is purely algorithmic
via vector fingerprinting or template matching.
"""

import io
import logging
import traceback
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.assemble import assemble_counts
from pipeline.classify import classify_clusters, render_legend_region
from pipeline.raster import (
    get_raster_positions_normalized,
    rasterize_page,
    run_template_matching,
)
from pipeline.triage import triage_pdf
from pipeline.vector import (
    extract_vector_clusters,
    get_cluster_positions,
    render_cluster_sample,
)

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
    2. Extract/detect symbols
    3. Classify via Claude (legend only)
    4. Assemble deterministic counts
    5. Return overlay data for verification UI

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

    # Read PDF bytes
    try:
        pdf_bytes = await pdf.read()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to read PDF: {e}")

    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="Empty PDF file")

    # Open PDF with PyMuPDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Corrupt or invalid PDF: {e}",
        )

    if len(doc) == 0:
        raise HTTPException(status_code=422, detail="PDF has no pages")

    try:
        # Stage 0 — Triage
        logger.info(f"Processing PDF: {pdf.filename}, {len(doc)} pages")
        page_routes = triage_pdf(doc)

        results: list = []

        for page_idx, route in page_routes:
            page = doc[page_idx]
            page_width = page.rect.width
            page_height = page.rect.height
            sheet_name = f"SHEET-{page_idx + 1}"

            logger.info(f"Processing {sheet_name} via '{route}' path")

            vector_positions = None
            raster_positions = None
            cluster_samples: Dict[str, bytes] = {}

            # Stage 1A — Vector Extraction
            if route in ("vector", "hybrid"):
                clusters = extract_vector_clusters(page)
                if clusters:
                    vector_positions = get_cluster_positions(
                        clusters, page_width, page_height
                    )
                    # Render first instance of each cluster for classification
                    for fp, instances in clusters.items():
                        if instances:
                            bbox = instances[0]["bbox"]
                            try:
                                sample_png = render_cluster_sample(page, bbox)
                                cluster_samples[fp] = sample_png
                            except Exception as e:
                                logger.warning(
                                    f"Failed to render sample for cluster {fp}: {e}"
                                )

            # Stage 1B — Template Matching (raster path)
            if route in ("raster", "hybrid"):
                # For raster path, we need templates
                # If we have vector clusters, use their rendered samples as templates
                if cluster_samples:
                    import cv2

                    templates: Dict[str, np.ndarray] = {}
                    for cid, png_bytes in cluster_samples.items():
                        img_array = np.frombuffer(png_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            templates[cid] = img

                    if templates:
                        matches = run_template_matching(page, templates)
                        # Get page dimensions in pixels at 300 DPI
                        dpi = 300
                        zoom = dpi / 72.0
                        px_width = int(page_width * zoom)
                        px_height = int(page_height * zoom)
                        raster_positions = get_raster_positions_normalized(
                            matches, px_width, px_height
                        )
                else:
                    logger.warning(
                        f"{sheet_name}: Raster path but no templates available. "
                        "Skipping template matching."
                    )

            # Stage 3 — AI Legend Classification
            classification = None
            if cluster_samples:
                try:
                    legend_png = render_legend_region(page)
                    classification = classify_clusters(legend_png, cluster_samples)
                except EnvironmentError as e:
                    # No API key — skip classification, use defaults
                    logger.warning(f"Classification skipped: {e}")
                    classification = {
                        "legend_entries": [],
                        "cluster_classification": [
                            {
                                "cluster_id": cid,
                                "classification": "sprinkler_head",
                                "head_type": "unknown",
                                "confidence": 0.5,
                            }
                            for cid in cluster_samples.keys()
                        ],
                    }
                except Exception as e:
                    logger.error(f"Classification failed: {e}")
                    classification = {
                        "legend_entries": [],
                        "cluster_classification": [
                            {
                                "cluster_id": cid,
                                "classification": "sprinkler_head",
                                "head_type": "unknown",
                                "confidence": 0.5,
                            }
                            for cid in cluster_samples.keys()
                        ],
                        "error": str(e),
                    }

            # Stage 4 — Count Assembly
            page_result = assemble_counts(
                vector_positions=vector_positions,
                raster_positions=raster_positions,
                classification=classification,
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

        response = {
            "filename": pdf.filename,
            "pages_processed": len(results),
            "total_heads": total_heads,
            "confidence": overall_confidence,
            "needs_verification": any_needs_verification,
            "sheets": results,
        }

        logger.info(
            f"Completed: {pdf.filename} — {total_heads} heads across "
            f"{len(results)} pages, confidence={overall_confidence:.2f}"
        )

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing failed: {str(e)}",
        )
    finally:
        doc.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
