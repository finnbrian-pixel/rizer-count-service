"""
Regression suite for Idaho FS-01 (ITD Maintenance Building).
Ground truth locked 2026-08-07. All assertions must pass before
any change to hc2.py, pipe.py, or physics.py is merged.

Ground truth (from live service run, authoritative):
  Total heads: 165
  By type: UPRIGHT K-8.0: 109, PENDENT K-5.6: 19, UPRIGHT K-5.6: 20, SIDEWALL K-5.6: 17
  Total pipe ft: 2009.1 (tolerance ±5 ft)
  Heads connected: 165 / 165
  Head coverage: 1.0
  Physics verdict: PLAUSIBLE
  needs_verification: false
  confidence (head): 1.0
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.regression


# --- Head count assertions ---

def test_total_head_count(count_result):
    assert count_result["total"] == 165, \
        f"Expected 165 heads, got {count_result['total']}"


def test_document_total_matches_page(count_result):
    """Document-level total should also be 165 (only one plan page)."""
    assert count_result["document_total"] == 165


def test_head_count_by_type(count_result):
    counts = count_result["counts"]
    assert counts.get("UPRIGHT K-8.0") == 109, \
        f"UPRIGHT K-8.0: expected 109, got {counts.get('UPRIGHT K-8.0')}"
    assert counts.get("PENDENT K-5.6") == 19, \
        f"PENDENT K-5.6: expected 19, got {counts.get('PENDENT K-5.6')}"
    assert counts.get("UPRIGHT K-5.6") == 20, \
        f"UPRIGHT K-5.6: expected 20, got {counts.get('UPRIGHT K-5.6')}"
    assert counts.get("SIDEWALL K-5.6") == 17, \
        f"SIDEWALL K-5.6: expected 17, got {counts.get('SIDEWALL K-5.6')}"


def test_no_extra_head_types(count_result):
    known = {"UPRIGHT K-8.0", "PENDENT K-5.6", "UPRIGHT K-5.6", "SIDEWALL K-5.6"}
    actual = set(count_result["counts"].keys())
    assert actual == known, f"Unexpected types: {actual - known}"


def test_type_counts_sum_to_total(count_result):
    """Sum of by-type counts must equal the total."""
    type_sum = sum(count_result["counts"].values())
    assert type_sum == count_result["total"], \
        f"Type sum {type_sum} != total {count_result['total']}"


def test_head_confidence(count_result):
    assert count_result["confidence"] == 1.0, \
        f"Expected confidence 1.0, got {count_result['confidence']}"


def test_needs_verification_false(count_result):
    assert count_result["needs_verification"] is False, \
        "needs_verification should be False for a validated 165-head count"


# --- Physics assertions ---

def test_physics_verdict(count_result):
    assert count_result["physics"]["verdict"] == "PLAUSIBLE", \
        f"Expected PLAUSIBLE, got {count_result['physics']['verdict']}"


def test_physics_confidence_multiplier(count_result):
    """PLAUSIBLE verdict has confidence_multiplier of 1.0."""
    assert count_result["physics"]["confidence_multiplier"] == 1.0


def test_physics_area_per_head(count_result):
    area = count_result["physics"]["implied_area_per_head_sqft"]
    assert 80 <= area <= 115, \
        f"Area per head {area} sqft outside expected band [80, 115]"


def test_physics_spacing_along_branch(count_result):
    d1 = count_result["physics"]["spacing_along_branch_ft"]
    assert 6.0 <= d1 <= 10.0, \
        f"Branch spacing {d1} ft out of range [6.0, 10.0]"


def test_physics_spacing_between_branches(count_result):
    d2 = count_result["physics"]["spacing_between_branches_ft"]
    assert 10.0 <= d2 <= 15.0, \
        f"Inter-branch spacing {d2} ft out of range [10.0, 15.0]"


def test_physics_needs_verification_false(count_result):
    assert count_result["physics"]["needs_verification"] is False


# --- Pipe assertions ---

def test_pipe_total_footage(pipe_result):
    assert abs(pipe_result["total_ft"] - 2009.1) <= 5.0, \
        f"Pipe total {pipe_result['total_ft']} differs from 2009.1 by more than ±5 ft"


def test_all_heads_connected(pipe_result):
    assert pipe_result["heads_connected"] == 165, \
        f"Expected 165 heads connected, got {pipe_result['heads_connected']}"


def test_head_coverage_full(pipe_result):
    assert pipe_result["head_coverage"] == 1.0, \
        f"Expected head_coverage 1.0, got {pipe_result['head_coverage']}"


def test_pipe_confidence(pipe_result):
    assert pipe_result["confidence"] == 1.0, \
        f"Expected pipe confidence 1.0, got {pipe_result['confidence']}"


def test_pipe_needs_verification_false(pipe_result):
    assert pipe_result["needs_verification"] is False


def test_pipe_dominant_size(pipe_result):
    """2" should be the largest single pipe size by footage."""
    by_size = pipe_result["by_size"]
    assert '2"' in by_size, f"2\" not in pipe sizes: {list(by_size.keys())}"
    largest = max(by_size, key=by_size.get)
    assert largest == '2"', f"Expected 2\" dominant, got {largest}"


def test_pipe_size_count(pipe_result):
    """Should have at least 4 distinct pipe sizes."""
    assert len(pipe_result["by_size"]) >= 4, \
        f"Only {len(pipe_result['by_size'])} sizes found: {list(pipe_result['by_size'].keys())}"


def test_pipe_no_excessive_unassigned(pipe_result):
    """Unassigned pipe footage should be < 15% of total."""
    total = pipe_result["total_ft"]
    unassigned = pipe_result["unassigned_ft"]
    if total > 0:
        ratio = unassigned / total
        assert ratio < 0.15, \
            f"Unassigned {unassigned:.1f} ft is {ratio:.1%} of total (> 15%)"


# --- Determinism assertions ---

def test_count_determinism(count_result):
    """Re-run and confirm the count is stable (no randomness)."""
    from conftest import FIXTURE_PATH
    import hc2
    r2 = hc2.count_document(FIXTURE_PATH, validate=True)
    assert r2["document_total"] == 165, \
        f"Non-deterministic: second run gave {r2['document_total']}"
