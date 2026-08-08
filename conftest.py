"""
Fixtures for Idaho FS-01 regression suite.
Ground truth locked 2026-08-07.

The fixture PDF must be placed at one of:
  1. fixtures/FM32361_ITB_FS01.pdf
  2. idaho.pdf (in repo root)

This is the Idaho Transportation Dept Maintenance Building fire sprinkler plan.
Expected output: 165 heads total (UPRIGHT K-8.0:109, PENDENT K-5.6:19,
UPRIGHT K-5.6:20, SIDEWALL K-5.6:17), 2009.1 ft pipe, PLAUSIBLE physics.
"""
import pytest
import sys
import os

# Ensure repo root is on path for hc2/pipe imports
sys.path.insert(0, os.path.dirname(__file__))

import hc2
import pipe as pipe_mod


def _find_fixture():
    """Locate the Idaho FS-01 fixture PDF. Returns path or None."""
    repo_root = os.path.dirname(__file__)
    candidates = [
        os.path.join(repo_root, "fixtures", "FM32361_ITB_FS01.pdf"),
        os.path.join(repo_root, "idaho.pdf"),
    ]
    for f in candidates:
        if os.path.isfile(f) and os.path.getsize(f) > 10000:
            # Must be >10KB to be a real sprinkler plan PDF, not a stub
            return f
    return None


FIXTURE_PATH = _find_fixture()

_SKIP_MSG = (
    "Idaho FS-01 fixture PDF not found (or too small). "
    "Place at fixtures/FM32361_ITB_FS01.pdf or idaho.pdf in repo root."
)


def pytest_collection_modifyitems(config, items):
    """Skip regression tests if the fixture PDF is not present."""
    if FIXTURE_PATH is not None:
        return
    skip_marker = pytest.mark.skip(reason=_SKIP_MSG)
    for item in items:
        if "regression" in item.keywords or "fs01" in item.nodeid.lower():
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def count_result():
    """Run hc2.count_document on the Idaho FS-01 fixture (validated)."""
    if FIXTURE_PATH is None:
        pytest.skip(_SKIP_MSG)
    result = hc2.count_document(FIXTURE_PATH, validate=True)
    # Idaho is a 2-page doc: page 0 = plan (165 heads), page 1 = details (0 heads)
    plan_page = None
    for p in result["pages"]:
        if p["sheet_kind"] == "plan" and p["total"] > 0:
            plan_page = p
            break
    assert plan_page is not None, "No plan page with heads found in Idaho FS-01"
    return {
        "total": plan_page["total"],
        "counts": plan_page["counts"],
        "confidence": plan_page["confidence"],
        "needs_verification": plan_page["needs_verification"],
        "physics": plan_page["physics"],
        "document_total": result["document_total"],
        "document_counts": result["document_counts"],
    }


@pytest.fixture(scope="session")
def pipe_result():
    """Run pipe.pipe_takeoff on the Idaho FS-01 fixture (page 0)."""
    if FIXTURE_PATH is None:
        pytest.skip(_SKIP_MSG)
    return pipe_mod.pipe_takeoff(FIXTURE_PATH, page_no=0)
