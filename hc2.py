"""
Legend-driven sprinkler head counting. No hardcoded symbol dimensions.

Every sprinkler plan carries its own decoder ring: the legend draws each symbol
at actual size next to its description. This module measures those swatches at
runtime and uses them as the match targets, so it works on drawing sets it has
never seen without tuning.
"""

import hashlib
import math
import re
from collections import defaultdict, Counter

import fitz


class _R:
    """Minimal rect: floats only, no fitz.Rect allocation."""
    __slots__ = ("x0", "y0", "x1", "y1", "width", "height")

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.width, self.height = x1 - x0, y1 - y0


SIZE_QUANT_PT = 0.5
ANGLE_QUANT_DEG = 5.0
CONCENTRIC_TOL_PT = 2.2
COINCIDENT_TOL_PT = 0.6

LEGEND_TITLE_PATTERNS = [
    r"SPRINKLER\s+SYMBOL\s+LEGEND",
    r"FIRE\s+SPRINKLER\s+LEGEND",
    r"SPRINKLER\s+LEGEND",
    r"SYMBOL\s+LEGEND",
    r"FIRE\s+PROTECTION\s+LEGEND",
    r"SPRINKLER\s+SYMBOLS?",
    r"LEGEND",
]

HEAD_WORDS = ("SPRINKLER", "PENDENT", "PENDANT", "UPRIGHT", "SIDEWALL", "CONCEALED")
NOT_HEAD_WORDS = ("RISER", "DEPARTMENT CONNECTION", "FDC", "DRAIN", "NODE",
                  "PIPING", "PIPE", "ELEVATION", "TRANSITION", "CAP", "VALVE",
                  "SWITCH", "GAUGE", "HANGER", "ZONE", "DUCT")


def _q(v, step=SIZE_QUANT_PT):
    return round(v / step) * step


def _qa(a):
    return round(a / ANGLE_QUANT_DEG) * ANGLE_QUANT_DEG % 180


def _sig_hash(sig_tuple):
    """Stable 8-char hex hash of a cluster signature tuple."""
    return hashlib.sha256(str(sig_tuple).encode()).hexdigest()[:8]


def element_of(d):
    """Build an element from a get_cdrawings() record (plain tuples, no
    Point/Rect objects -- roughly half the memory of get_drawings())."""
    x0, y0, x1, y1 = d["rect"]
    r = _R(x0, y0, x1, y1)
    angles = []
    for it in d["items"]:
        if it[0] == "l":
            p1, p2 = it[1], it[2]
            angles.append(_qa(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180))
    return {
        "cx": (r.x0 + r.x1) / 2.0,
        "cy": (r.y0 + r.y1) / 2.0,
        "kinds": "".join(sorted(it[0] for it in d["items"])),
        "w": _q(r.width),
        "h": _q(r.height),
        "raw_w": r.width,
        "raw_h": r.height,
        "filled": d.get("fill") is not None,
        "angles": tuple(sorted(angles)),
    }


def descriptor(e):
    """Position-independent description of one drawn element."""
    return (e["kinds"], e["w"], e["h"], e["filled"], e["angles"])


USE_STREAM = True   # parse the content stream instead of materializing paths


def _boxes(page, min_w, min_h):
    """Larger drawn rectangles, streamed."""
    if USE_STREAM:
        try:
            import stream_extract
            return list(stream_extract.stream_boxes(page, min_w, min_h))
        except Exception:
            pass
    return [tuple(d["rect"]) for d in page.get_cdrawings()
            if d["rect"][2] - d["rect"][0] >= min_w
            and d["rect"][3] - d["rect"][1] >= min_h]


def small_elements(page, max_pt=30.0):
    """Small drawn elements only.

    Default path parses the raw content stream incrementally, discarding
    oversized paths as soon as their bbox is known. On a dense Revit export
    (1.2M paths) that is ~123 MB peak instead of ~1.8 GB for get_cdrawings().
    """
    if USE_STREAM:
        try:
            import stream_extract
            return list(stream_extract.stream_elements(page, max_pt=max_pt))
        except Exception:
            pass   # fall back to the materializing path
    out = []
    for d in page.get_cdrawings():
        x0, y0, x1, y1 = d["rect"]
        if 0.5 < x1 - x0 < max_pt and 0.5 < y1 - y0 < max_pt:
            out.append(element_of(d))
    return out


def build_index(elements, cell=None):
    cell = cell or CONCENTRIC_TOL_PT * 2
    idx = defaultdict(list)
    for e in elements:
        idx[(int(e["cx"] // cell), int(e["cy"] // cell))].append(e)
    return idx, cell


def companions(core, index, cell):
    """Non-transitive: elements concentric with THIS core only. No chaining."""
    out = []
    gx, gy = int(core["cx"] // cell), int(core["cy"] // cell)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for e in index.get((gx + dx, gy + dy), []):
                if e is core:
                    continue
                if (abs(e["cx"] - core["cx"]) < CONCENTRIC_TOL_PT
                        and abs(e["cy"] - core["cy"]) < CONCENTRIC_TOL_PT):
                    out.append(e)
    return out


def collapse(elems, tol=COINCIDENT_TOL_PT):
    """Drop elements that duplicate another at the same center (double strokes)."""
    out = []
    for e in elems:
        if not any(descriptor(e) == descriptor(o)
                   and abs(e["cx"] - o["cx"]) < tol
                   and abs(e["cy"] - o["cy"]) < tol for o in out):
            out.append(e)
    return out


def cluster_signature(seed, index, cell, noise_floor=0.0):
    """Signature of the whole concentric cluster containing `seed`.

    Matching whole clusters (not individual elements) counts each symbol once
    regardless of how many primitives it is drawn with, and makes the legend
    swatch and the plan instance produce the identical key.

    noise_floor drops artifacts that merely overlap the symbol -- dashed-line
    dots and hatching leave 1pt specks inside sidewall triangles, which would
    otherwise corrupt the signature.
    """
    members = [m for m in [seed] + companions(seed, index, cell)
               if max(m["raw_w"], m["raw_h"]) >= noise_floor or m is seed]
    members = collapse(members)
    cx = sum(m["cx"] for m in members) / len(members)
    cy = sum(m["cy"] for m in members) / len(members)
    return tuple(sorted(descriptor(m) for m in members)), cx, cy


# ------------------------------------------------------------ legend read ---

def find_legend(page):
    """Locate the legend table without knowing its exact title."""
    text = page.get_text()
    for pat in LEGEND_TITLE_PATTERNS:
        m = re.search(pat, text, re.I)
        if not m:
            continue
        for hit in page.search_for(m.group(0)):
            table = _table_box(page, hit)
            rows = _rows_in(page, table)
            if len(rows) >= 2:
                return hit, table, rows
    return None, None, []


def _table_box(page, hit):
    """Bound the legend by the rectangle drawn around it.

    A fixed-height window swallows whatever sits below the legend -- general
    notes, schedules -- and their text hijacks the description-column
    detection. Most legends are boxed; use that box when present.
    """
    page_area = page.rect.width * page.rect.height
    title_w = hit.x1 - hit.x0
    best = None
    for x0, y0, x1, y1 in _boxes(page, 80, 30):
        w, h = x1 - x0, y1 - y0
        # sanity: a legend box is not the sheet border
        if w > title_w * 3.5 or h > 520 or w * h > page_area * 0.10:
            continue
        if not (x0 - 6 <= hit.x0 and x1 + 6 >= hit.x1
                and y0 - 6 <= hit.y0 and y1 + 6 >= hit.y1):
            continue
        area = w * h
        if best is None or area < best[0]:
            best = (area, fitz.Rect(x0, y0, x1, y1))
    if best:
        return best[1]
    # no enclosing box: stop at the first large vertical gap below the title
    return fitz.Rect(hit.x0 - 30, hit.y1,
                     min(hit.x1 + 140, page.rect.x1),
                     min(hit.y1 + 260, page.rect.y1))


def _rows_in(page, table):
    """Derive legend rows. The description column is found by histogram of
    text left-edges (descriptions share one left margin); the symbol column is
    everything left of it, inside the table."""
    lines = []
    for b in page.get_text("dict", clip=table)["blocks"]:
        for l in b.get("lines", []):
            txt = "".join(s["text"] for s in l["spans"]).strip()
            if txt and txt.upper() not in ("SYMBOL", "DESCRIPTION", "SYMBOLS"):
                lines.append((l["bbox"][1], l["bbox"][3], l["bbox"][0], txt))
    if not lines:
        return []
    hist = Counter(round(l[2]) for l in lines)
    desc_x = float(max(hist.items(), key=lambda kv: (kv[1], -kv[0]))[0])

    desc_lines = sorted(l for l in lines if l[2] >= desc_x - 3)
    bands, cur = [], []
    for ln in desc_lines:
        if cur and ln[0] - cur[-1][1] > 4.0:
            bands.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        bands.append(cur)

    return [{"y0": min(l[0] for l in b) - 4,
             "y1": max(l[1] for l in b) + 4,
             "desc_x": desc_x,
             "sym_x0": table.x0,
             "description": " ".join(l[3] for l in b)} for b in bands]


DETAIL_RE = re.compile(r"\bDETAIL\b", re.I)


def detail_zones(page):
    """Boxed detail callouts drawn on a plan sheet.

    Details show sample heads at plan scale. They are not installed heads and
    must not be counted; a boxed region labelled DETAIL is the reliable marker.
    """
    zones = []
    page_area = page.rect.width * page.rect.height
    labels = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            txt = "".join(s["text"] for s in l["spans"])
            if DETAIL_RE.search(txt):
                labels.append(fitz.Rect(l["bbox"]))
    if not labels:
        return zones
    for x0, y0, x1, y1 in _boxes(page, 60, 40):
        w, h = x1 - x0, y1 - y0
        if w * h > page_area * 0.10:
            continue
        r = fitz.Rect(x0, y0, x1, y1)
        for lab in labels:
            if r.x0 - 6 <= lab.x0 and r.x1 + 6 >= lab.x1 and r.y0 - 6 <= lab.y0 and r.y1 + 6 >= lab.y1:
                zones.append(r)
                break
    # keep only the smallest box around each label
    keep = []
    for z in sorted(zones, key=lambda r: r.get_area()):
        if not any(k.contains(z) for k in keep):
            keep.append(z)
    return keep


def is_head_row(desc):
    d = desc.upper()
    if any(k in d for k in NOT_HEAD_WORDS):
        return False
    return any(k in d for k in HEAD_WORDS)


def learn_symbols(page, table, rows, elements):
    """MEASURE each legend swatch. This replaces every hardcoded constant.

    For each head row, take the drawn elements in the symbol column and
    register a signature for every element treated as core. Registering all
    of them removes the legend-vs-plan 'which element is the core' mismatch.

    Returns (learned, base_sigs) where learned maps every variant signature
    to its label and base_sigs is the set of signatures that matched without
    any rotation/mirror (the canonical legend orientation).
    """
    learned = {}
    base_sigs = set()
    for row in rows:
        if not is_head_row(row["description"]):
            continue
        parts = [e for e in elements
                 if row["y0"] < e["cy"] < row["y1"]
                 and row["sym_x0"] < e["cx"] < row["desc_x"] - 2]
        if not parts:
            continue
        idx, cell = build_index(parts)
        label = short_label(row["description"])
        sig = tuple(sorted(descriptor(p) for p in collapse(parts)))
        base_sigs.add(sig)
        for variant in orientation_variants(sig):
            learned.setdefault(variant, label)
        row["measured"] = [(p["kinds"], round(p["raw_w"], 1), round(p["raw_h"], 1),
                            p["filled"], p["angles"]) for p in parts]
    return learned, base_sigs


def _rot_desc(d, k, mirror):
    """Rotate/mirror one descriptor. Directional symbols (sidewalls, drops)
    are drawn once in the legend but appear at any orientation on the plan."""
    kinds, w, h, filled, angles = d
    a = list(angles)
    if mirror:
        a = [(180.0 - x) % 180 for x in a]
    a = [(x + 90.0 * k) % 180 for x in a]
    if k % 2 == 1:
        w, h = h, w
    return (kinds, w, h, filled, tuple(sorted(_qa(x) for x in a)))


def orientation_variants(sig):
    """All 8 rigid orientations of a swatch signature."""
    out = set()
    for k in range(4):
        for m in (False, True):
            out.add(tuple(sorted(_rot_desc(d, k, m) for d in sig)))
    return out


def short_label(description):
    d = description.upper()
    kind = ("DRY PENDENT" if "DRY PENDENT" in d or "DRY PENDANT" in d else
            "DRY SIDEWALL" if "DRY SIDEWALL" in d else
            "DRY UPRIGHT" if "DRY UPRIGHT" in d else
            "CONCEALED" if "CONCEALED" in d else
            "PENDENT" if "PENDENT" in d or "PENDANT" in d else
            "SIDEWALL" if "SIDEWALL" in d else
            "UPRIGHT" if "UPRIGHT" in d else "SPRINKLER")
    m = re.search(r"K[- ]?(\d+\.\d+)", d)
    return f"{kind} K-{m.group(1)}" if m else kind


# ---------------------------------------------------------- sheet triage ----

def sheet_kind_from_result(n_heads, has_legend):
    """Content-based triage: a plan sheet is one where the legend's own symbols
    actually appear outside the legend. Title text varies too much between
    engineers to rely on ('FIRST FLOOR SPRINKLER PLAN' vs 'PLAN - MAIN FLOOR').
    A legend-only or details sheet yields zero matches and says so."""
    if n_heads > 0:
        return "plan"
    return "legend_or_details" if has_legend else "unknown"


# -------------------------------------------------------------- counting ----

def dedupe_coincident(items, tol=COINCIDENT_TOL_PT):
    grid, out = defaultdict(list), []
    cell = max(tol * 2, 1.0)
    for it in items:
        gx, gy = int(it["cx"] // cell), int(it["cy"] // cell)
        dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for o in grid.get((gx + dx, gy + dy), []):
                    if abs(it["cx"] - o["cx"]) < tol and abs(it["cy"] - o["cy"]) < tol:
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            out.append(it)
            grid[(gx, gy)].append(it)
    return out


def count_page(pdf_path, page_no=0, extra_symbols=None, extra_base_sigs=None):
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    elements = small_elements(page)
    anchor, table, rows = find_legend(page)
    if table:
        learned, base_sigs = learn_symbols(page, table, rows, elements)
    else:
        learned, base_sigs = {}, set()
    if extra_symbols:
        for sig, label in extra_symbols.items():
            learned.setdefault(sig, label)
    if extra_base_sigs:
        base_sigs.update(extra_base_sigs)

    # Exclude the legend TABLE RECTANGLE, not a half-plane: legends appear on
    # the left as often as the right, and a half-plane rule wipes the sheet.
    zones = []
    if table:
        rows_y1 = max((r["y1"] for r in rows), default=table.y1)
        zones.append(fitz.Rect(table.x0 - 8, min(table.y0, anchor.y0) - 8,
                               table.x1 + 8, max(rows_y1, table.y0) + 8))
    zones.extend(detail_zones(page))

    def excluded(e):
        return any(z.x0 <= e["cx"] <= z.x1 and z.y0 <= e["cy"] <= z.y1 for z in zones)

    result = {
        "page": page_no,
        "learned_types": sorted(set(learned.values())),
        "learned_signatures": len(learned),
        "excluded_zones": [[round(v, 1) for v in (z.x0, z.y0, z.x1, z.y1)] for z in zones],
    }

    if not learned:
        result["sheet_kind"] = sheet_kind_from_result(0, table is not None)
        result.update({"total": 0, "counts": {}, "heads": [],
                       "confidence": 0.0, "needs_verification": True,
                       "reason": "no head symbols learned from legend"})
        doc.close()
        return result

    idx, cell = build_index(elements)
    noise_floor = 0.5 * min(
        (min(p[1], p[2]) for r in rows if r.get("measured") for p in r["measured"]),
        default=0.0)
    result["noise_floor"] = round(noise_floor, 2)
    found = []
    for e in elements:
        if excluded(e):
            continue
        if max(e["raw_w"], e["raw_h"]) < noise_floor:
            continue
        sig, cx, cy = cluster_signature(e, idx, cell, noise_floor)
        if sig in learned:
            # Per-detection confidence: base signature = 0.95, orientation variant = 0.85
            conf = 0.95 if sig in base_sigs else 0.85
            found.append({"cx": cx, "cy": cy, "type": learned[sig], "sig_hash": _sig_hash(sig), "confidence": conf})

    heads = [{"x": round(f["cx"], 1), "y": round(f["cy"], 1),
             "type": f["type"], "signature_hash": f["sig_hash"],
             "confidence": f.get("confidence", 0.95)}
             for f in dedupe_coincident(found, tol=CONCENTRIC_TOL_PT)]

    result.update({
        "sheet_kind": sheet_kind_from_result(len(heads), table is not None),
        "total": len(heads),
        "counts": dict(Counter(h["type"] for h in heads)),
        "heads": heads,
        "confidence": 1.0 if heads else 0.0,
        "needs_verification": len(heads) == 0,
    })
    doc.close()
    return result


def count_page_validated(pdf_path, page_no=0, hazard=None):
    """Count, then validate against the drawing's own declared physics.

    Confidence is the product of the geometric result and the physics verdict,
    so an implausible count cannot come back looking trustworthy.
    """
    import physics
    r = count_page(pdf_path, page_no)
    p = physics.check(pdf_path, r, page_no, hazard)
    r["physics"] = p
    r["confidence"] = round(r.get("confidence", 0.0) * p["confidence_multiplier"], 3)
    r["needs_verification"] = r.get("needs_verification", False) or p["needs_verification"]
    return r


def learn_document_symbols(pdf_path):
    """Pass 1: pool head symbols from every sheet that carries a legend.

    Drawing sets routinely put the legend on its own sheet (F001) and the
    plans on others (F101, F102). Counting each sheet in isolation makes those
    plan sheets return zero -- they have nothing to learn from.
    """
    doc = fitz.open(pdf_path)
    pooled, pooled_base_sigs, sources = {}, set(), []
    for i in range(len(doc)):
        page = doc[i]
        elements = small_elements(page)
        anchor, table, rows = find_legend(page)
        if not table:
            continue
        learned, base_sigs = learn_symbols(page, table, rows, elements)
        if learned:
            for sig, label in learned.items():
                pooled.setdefault(sig, label)
            pooled_base_sigs.update(base_sigs)
            sources.append({"page": i, "types": sorted(set(learned.values()))})
    doc.close()
    return pooled, pooled_base_sigs, sources


def count_document(pdf_path, validate=True):
    """Two-pass: learn symbols across the whole set, then count each sheet."""
    pooled, pooled_base_sigs, sources = learn_document_symbols(pdf_path)
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()

    pages = []
    for i in range(n):
        p = count_page(pdf_path, i, extra_symbols=pooled, extra_base_sigs=pooled_base_sigs)
        if validate and p["total"]:
            try:
                import physics
                ph = physics.check(pdf_path, p, i)
                p["physics"] = ph
                p["confidence"] = round(p["confidence"] * ph["confidence_multiplier"], 3)
                p["needs_verification"] = p["needs_verification"] or ph["needs_verification"]
            except Exception as e:
                p["physics"] = {"verdict": "UNKNOWN", "error": str(e)}
        pages.append(p)

    agg = Counter()
    for p in pages:
        agg.update(p["counts"])
    plan_pages = [p for p in pages if p["sheet_kind"] == "plan"]
    return {
        "file": pdf_path,
        "sheets": n,
        "legend_sources": sources,
        "pooled_symbol_types": sorted(set(pooled.values())),
        "pages": pages,
        "plan_sheets": [p["page"] for p in plan_pages],
        "document_total": sum(p["total"] for p in pages),
        "document_counts": dict(agg),
        "needs_verification": any(p["needs_verification"] for p in plan_pages) or not plan_pages,
    }


def count_in_region(result, x0, y0, x1, y1):
    return sum(1 for h in result["heads"] if x0 < h["x"] < x1 and y0 < h["y"] < y1)


if __name__ == "__main__":
    import sys, json
    r = count_document(sys.argv[1])
    for p in r["pages"]:
        p.pop("heads", None)
    print(json.dumps(r, indent=2, default=str))
