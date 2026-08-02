"""
NFPA physics self-check for sprinkler head counts.

A count can be geometrically clean and still be wrong. This module validates a
count against the physics the drawing itself declares: sheet scale, hazard
classification, and NFPA 13 maximum coverage area per sprinkler.

Heads sit on branch lines at regular spacing because the code requires it. That
makes density a hard constraint, not a heuristic -- a count that implies 7 sq ft
per head is wrong no matter how tidy the geometry looked.

Catches, without a human:
  3,832 on Idaho FS-01  -> ~4 sq ft/head   -> IMPLAUSIBLE
    690 on Idaho FS-01  -> ~23 sq ft/head  -> IMPLAUSIBLE
    165 on Idaho FS-01  -> ~96 sq ft/head  -> PLAUSIBLE (limit 130)
"""

import re
import math
import statistics
from collections import Counter

import fitz


NFPA_DEFAULT_COVERAGE = {
    "light": 225.0,
    "ordinary": 130.0,
    "extra": 100.0,
}

NFPA_MAX_SPACING = {
    "light": 15.0,
    "ordinary": 15.0,
    "extra": 12.0,
}

ABSOLUTE_MIN_SPACING_FT = 4.0


_SCALE_RE = re.compile(
    r'(\d+)\s*/\s*(\d+)\s*[\u201c\u201d\u2018\u2019"\'\\]\s*=\s*(\d+)\s*[\u2018\u2019\'\\]\s*-?\s*(\d+)?'
)


def parse_scale(page):
    best = None
    for m in _SCALE_RE.finditer(page.get_text()):
        num, den, ft, inch = m.groups()
        paper_in = float(num) / float(den)
        world_ft = float(ft) + (float(inch or 0) / 12.0)
        if paper_in <= 0 or world_ft <= 0:
            continue
        ft_per_inch = world_ft / paper_in
        cand = ft_per_inch / 72.0
        if best is None or cand == best:
            best = cand
    return best


def parse_coverage_limits(doc):
    limits = dict(NFPA_DEFAULT_COVERAGE)
    declared = {}
    for page in doc:
        txt = page.get_text().upper()
        for m in re.finditer(
            r"(LIGHT|ORDINARY|EXTRA)\s+HAZARD\s*:?\s*(\d{2,4})\s*SQ\s*\.?\s*FT", txt
        ):
            declared[m.group(1).lower()] = float(m.group(2))
    limits.update(declared)
    return limits, declared


def parse_hazard_classes(doc):
    found = []
    for page in doc:
        txt = page.get_text().upper()
        for cls in ("EXTRA", "ORDINARY", "LIGHT"):
            if re.search(cls + r"\s+HAZARD", txt) and cls.lower() not in found:
                found.append(cls.lower())
    return found or ["ordinary"]


def _nn_stats(points):
    if len(points) < 4:
        return None, None
    d1, d2 = [], []
    for i, (x, y) in enumerate(points):
        ds = sorted(math.hypot(x - px, y - py)
                    for j, (px, py) in enumerate(points) if j != i)
        ds = [d for d in ds if d > 0.01]
        if not ds:
            continue
        d1.append(ds[0])
        for d in ds:
            if d > ds[0] * 1.3:
                d2.append(d)
                break
    if not d1:
        return None, None
    return (statistics.median(d1),
            statistics.median(d2) if d2 else statistics.median(d1))


CONFIDENCE = {
    "PLAUSIBLE": 1.0,
    "SUSPECT_UNDERCOUNT": 0.6,
    "IMPLAUSIBLE": 0.0,
    "NO_HEADS": 0.0,
    "TOO_FEW_TO_VALIDATE": 0.8,
    "UNKNOWN": 0.8,
}


def _finalize(out):
    out["confidence_multiplier"] = CONFIDENCE.get(out["verdict"], 0.5)
    out["needs_verification"] = out["verdict"] != "PLAUSIBLE"
    return out


def check(pdf_path, result, page_no=0, hazard=None):
    doc = fitz.open(pdf_path)
    page = doc[page_no]

    ft_per_pt = parse_scale(page)
    limits, declared = parse_coverage_limits(doc)
    classes = parse_hazard_classes(doc)
    hazard = hazard or ("ordinary" if "ordinary" in classes else classes[0])
    max_cov = limits[hazard]
    max_space = NFPA_MAX_SPACING[hazard]
    doc.close()

    heads = result.get("heads", [])
    out = {
        "scale_ft_per_pt": round(ft_per_pt, 5) if ft_per_pt else None,
        "scale_note": ("1 pt = %.3f ft" % ft_per_pt) if ft_per_pt else "scale not found",
        "hazard_classes_on_sheet": classes,
        "hazard_used": hazard,
        "max_coverage_sqft": max_cov,
        "coverage_source": "drawing notes" if declared else "NFPA 13 default",
        "head_count": len(heads),
        "verdict": "UNKNOWN",
        "flags": [],
    }

    if not heads:
        out["verdict"] = "NO_HEADS"
        out["flags"].append("zero heads counted; nothing to validate")
        return _finalize(out)
    if ft_per_pt is None:
        out["flags"].append("sheet scale not found; cannot validate density")
        return _finalize(out)
    if len(heads) < 4:
        out["verdict"] = "TOO_FEW_TO_VALIDATE"
        return _finalize(out)

    pts = [(h["x"], h["y"]) for h in heads]
    d1_pt, d2_pt = _nn_stats(pts)
    if d1_pt is None:
        out["flags"].append("could not derive spacing")
        return _finalize(out)

    d1_ft, d2_ft = d1_pt * ft_per_pt, d2_pt * ft_per_pt
    area_per_head = d1_ft * d2_ft

    out.update({
        "spacing_along_branch_ft": round(d1_ft, 1),
        "spacing_between_branches_ft": round(d2_ft, 1),
        "implied_area_per_head_sqft": round(area_per_head, 1),
        "max_spacing_allowed_ft": max_space,
    })

    if d1_ft < ABSOLUTE_MIN_SPACING_FT:
        out["verdict"] = "IMPLAUSIBLE"
        out["flags"].append(
            f"heads {d1_ft:.1f} ft apart; no hazard class permits under "
            f"{ABSOLUTE_MIN_SPACING_FT:.0f} ft -- almost certainly an over-count")
    elif area_per_head < max_cov * 0.25:
        out["verdict"] = "IMPLAUSIBLE"
        out["flags"].append(
            f"{area_per_head:.0f} sq ft per head vs {max_cov:.0f} allowed; "
            f"over-designed by {max_cov/area_per_head:.1f}x -- likely over-count")
    elif area_per_head > max_cov:
        out["verdict"] = "SUSPECT_UNDERCOUNT"
        out["flags"].append(
            f"{area_per_head:.0f} sq ft per head exceeds the {max_cov:.0f} sq ft "
            f"limit for {hazard} hazard -- heads may have been missed")
    elif max(d1_ft, d2_ft) > max_space:
        out["verdict"] = "SUSPECT_UNDERCOUNT"
        out["flags"].append(
            f"spacing {max(d1_ft, d2_ft):.1f} ft exceeds the {max_space:.0f} ft "
            f"limit for {hazard} hazard")
    else:
        out["verdict"] = "PLAUSIBLE"
        out["flags"].append(
            f"{area_per_head:.0f} sq ft per head, within the {max_cov:.0f} sq ft "
            f"{hazard}-hazard limit; spacing {d1_ft:.1f} x {d2_ft:.1f} ft")

    return _finalize(out)
