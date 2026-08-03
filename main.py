"""
RIZER count service — FastAPI.

Reference implementation. Everything the frontend needs comes back in one
response, already resolved: which sheet is the plan, the detections for it, and
the document totals. The frontend should never have to decide which page to
show or reconcile per-sheet with job totals.

Design rules this file follows, each one from a bug that actually happened:

  * Lazy imports. cv2/numpy at module scope cost ~140 MB resident before a
    single request. They are imported inside the functions that need them.
  * Uploads stream to a temp file, never held in memory.
  * Every page is reported, but `active_sheet` names the plan sheet. A details
    sheet legitimately returns 0 heads; that is not an error and must not be
    surfaced as one.
  * A count of 0 with no plan sheet returns 200 with needs_verification, not a
    500. "Rejected" is a valid outcome, not a failure.
  * Pipe is behind `include_pipe`, off by default, until hand-verified.
  * RSS logged per stage so a memory regression is visible in the logs.

Run:  uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
"""

import logging
import os
import shutil
import tempfile
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("count-service")

ENGINE_VERSION = "hc2-2026.08.03"
MAX_UPLOAD_MB = 60

app = FastAPI(title="RIZER Count Service", version=ENGINE_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cfpdesignai.netlify.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rss_mb():
    """Resident memory in MB. Logged at each stage so regressions are visible."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024)
    except Exception:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        return -1


@app.get("/health")
def health():
    return {"ok": True, "engine": ENGINE_VERSION, "rss_mb": rss_mb()}


@app.post("/count")
async def count(
    file: UploadFile = File(...),
    include_pipe: bool = Form(False),
):
    run_id = str(uuid.uuid4())
    started = time.time()
    log.info("run %s: upload begin, rss %s MB", run_id, rss_mb())

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported by this endpoint")

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"PDF exceeds {MAX_UPLOAD_MB} MB")
            tmp.write(chunk)
        tmp.close()
        log.info("run %s: upload done, %.1f MB file, rss %s MB",
                 run_id, size / 1e6, rss_mb())

        payload = _process(tmp.name, file.filename, include_pipe, run_id)
        payload["runtime_ms"] = int((time.time() - started) * 1000)
        payload["peak_mb"] = rss_mb()
        log.info("run %s: done in %s ms, rss %s MB, total %s heads on sheet %s",
                 run_id, payload["runtime_ms"], payload["peak_mb"],
                 payload["document"]["total_heads"], payload["active_sheet"])
        return payload
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _process(path, original_name, include_pipe, run_id):
    import hc2                      # lazy: keeps baseline RSS low

    doc = hc2.count_document(path)
    log.info("run %s: %s sheets, plan sheets %s, rss %s MB",
             run_id, doc["sheets"], doc["plan_sheets"], rss_mb())

    sheets = []
    for page in doc["pages"]:
        sheets.append({
            "page": page["page"],
            "sheet_kind": page["sheet_kind"],
            "is_plan": page["sheet_kind"] == "plan",
            "total": page["total"],
            "counts": page["counts"],
            "learned_types": page["learned_types"],
            "confidence": page["confidence"],
            "needs_verification": page["needs_verification"],
            "physics": page.get("physics"),
            "reason": page.get("reason"),
        })

    # The single most important line in this file: pick the plan sheet.
    # A details sheet correctly returns 0 and must not be shown as the result.
    active = doc["plan_sheets"][0] if doc["plan_sheets"] else None

    detections = []
    if active is not None:
        page = doc["pages"][active]
        detections = [
            {
                "id": f"{run_id}:{active}:{i}",
                "cx": h["x"],
                "cy": h["y"],
                "classification": h["type"],
                "confidence": page["confidence"],
                "signature_hash": h.get("signature_hash"),
            }
            for i, h in enumerate(page["heads"])
        ]

    geom = _page_geometry(path, active)

    payload = {
        "run_id": run_id,
        "engine_version": ENGINE_VERSION,
        "filename": original_name,
        "active_sheet": active,
        "sheets": sheets,
        "detections": detections,
        "document": {
            "total_heads": doc["document_total"],
            "counts": doc["document_counts"],
            "sheet_count": doc["sheets"],
            "plan_sheets": doc["plan_sheets"],
            "needs_verification": doc["needs_verification"],
        },
        "page": geom,
        "pipe": None,
    }

    if active is None:
        payload["notice"] = (
            "No sheet in this set contains a sprinkler head layout. Design-intent "
            "and permit sets often show hazard areas only; the contractor lays out "
            "the heads. Nothing to count."
        )

    if include_pipe and active is not None:
        payload["pipe"] = _pipe(path, active, run_id)

    return payload


def _page_geometry(path, active):
    """Page size in PDF points — the coordinate space detections are reported in."""
    if active is None:
        return None
    import fitz
    doc = fitz.open(path)
    try:
        page = doc[active]
        return {"page_no": active,
                "width": page.rect.width,
                "height": page.rect.height}
    finally:
        doc.close()


def _pipe(path, active, run_id):
    try:
        import pipe as pipe_mod
        result = pipe_mod.pipe_takeoff(path, active)
        log.info("run %s: pipe %.0f ft, coverage %s, rss %s MB",
                 run_id, result["total_ft"], result.get("head_coverage"), rss_mb())
        return result
    except Exception as exc:                      # never fail the count over pipe
        log.exception("run %s: pipe takeoff failed", run_id)
        return {"error": str(exc), "needs_verification": True}
