"""
Legend-driven sprinkler head counting. No hardcoded symbol dimensions.

Every sprinkler plan carries its own decoder ring: the legend draws each symbol
at actual size next to its description. This module measures those swatches at
runtime and uses them as the match targets, so it works on drawing sets it has
never seen without tuning.
"""

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


def small_elements(page, max_pt=30.0):
    """Stream drawings; keep only small ones. Never materialize the full list.

    page.get_drawings() on a dense sheet returns ~92k dicts. Binding that to a
    name (or calling it twice) is what put this over a 512MB container.
    """
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
            table = fitz.Rect(hit.x0 - 30, hit.y1,
                              min(hit.x1 + 120, page.rect.x1),
                              min(hit.y1 + 700, page.rect.y1))
            rows = _rows_in(page, table)
            if len(rows) >= 2:
                return hit, table, rows
    return None, None, []


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
    """
    learned = {}
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
        for variant in orientation_variants(sig):
            learned.setdefault(variant, label)
        row["measured"] = [(p["kinds"], round(p["raw_w"], 1), round(p["raw_h"], 1),
                            p["filled"], p["angles"]) for p in parts]
    return learned


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


def count_page(pdf_path, page_no=0):
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    elements = small_elements(page)
    anchor, table, rows = find_legend(page)
    learned = learn_symbols(page, table, rows, elements) if table else {}

    # Exclude the legend TABLE RECTANGLE, not a half-plane: legends appear on
    # the left as often as the right, and a half-plane rule wipes the sheet.
    zones = []
    if table:
        rows_y1 = max((r["y1"] for r in rows), default=table.y1)
        zones.append(fitz.Rect(table.x0 - 8, min(table.y0, anchor.y0) - 8,
                               table.x1 + 8, max(rows_y1, table.y0) + 8))

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
            found.append({"cx": cx, "cy": cy, "type": learned[sig]})

    heads = [{"x": round(f["cx"], 1), "y": round(f["cy"], 1), "type": f["type"]}
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


def count_document(pdf_path):
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    pages = [count_page(pdf_path, i) for i in range(n)]
    total = sum(p["total"] for p in pages)
    agg = Counter()
    for p in pages:
        agg.update(p["counts"])
    return {"file": pdf_path, "pages": pages, "document_total": total,
            "document_counts": dict(agg)}


def count_in_region(result, x0, y0, x1, y1):
    return sum(1 for h in result["heads"] if x0 < h["x"] < x1 and y0 < h["y"] < y1)


if __name__ == "__main__":
    import sys, json
    r = count_document(sys.argv[1])
    for p in r["pages"]:
        p.pop("heads", None)
    print(json.dumps(r, indent=2, default=str))
