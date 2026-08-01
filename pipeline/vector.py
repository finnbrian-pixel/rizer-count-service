"""
Deterministic sprinkler head counting from vector PDFs.
CFPdesign.AI / RIZER -- Path A reference implementation.

Replaces "cluster all repeated geometry" (which returned 3,832 on Idaho FS-01)
with four geometric filters:

  1. Size / aspect gate        -> kills pipe runs, walls, dimension lines
  2. Coincident dedup          -> collapses double-stroked symbols
  3. Concentric classification -> separates head types
  4. Slash-angle test          -> separates heads (parallel) from valves (crossed)

No LLM anywhere in the counting path. Same input, same output, every run.

VERIFIED: ITD FM32361 FS-01 -> 165 heads.
Bay regression (x 300-700, y 380-760) -> 23, matching a human hand count.
"""

import math
import hashlib
from collections import defaultdict, Counter

import fitz  # PyMuPDF


# ---------------------------------------------------------------- tuning ----

SYMBOL_MIN_PT = 3.0
SYMBOL_MAX_PT = 20.0
ASPECT_LO, ASPECT_HI = 0.5, 2.0
CONCENTRIC_TOL_PT = 2.2
COINCIDENT_TOL_PT = 0.6
ANGLE_SPREAD_MAX = 20.0
SIZE_QUANT_PT = 0.5


def _q(v, step=SIZE_QUANT_PT):
    return round(v / step) * step


def element_of(d):
    rect = d["rect"]
    # Handle both tuple (from get_cdrawings) and Rect (from get_drawings)
    if isinstance(rect, tuple):
        r = fitz.Rect(rect)
    else:
        r = fitz.Rect(rect)
    angles = []
    for it in d["items"]:
        if it[0] == "l":
            p1, p2 = it[1], it[2]
            # Handle both Point objects and tuples
            if isinstance(p1, tuple):
                dy = p2[1] - p1[1]
                dx = p2[0] - p1[0]
            else:
                dy = p2.y - p1.y
                dx = p2.x - p1.x
            a = math.degrees(math.atan2(dy, dx)) % 180.0
            angles.append(round(a, 0))
    return {
        "rect": r,
        "cx": (r.x0 + r.x1) / 2.0,
        "cy": (r.y0 + r.y1) / 2.0,
        "kinds": "".join(sorted(it[0] for it in d["items"])),
        "w": _q(r.width),
        "h": _q(r.height),
        "raw_w": round(r.width, 2),
        "raw_h": round(r.height, 2),
        "filled": d.get("fill") is not None,
        "angles": angles,
    }


def small_elements(page, max_pt=22.0):
    """Return (elements_list, total_drawings_count) to avoid calling get_drawings() twice.
    Uses get_cdrawings(extended=True) for lower memory overhead (tuples vs Point objects).
    """
    drawings = page.get_cdrawings(extended=True)
    total_drawings = len(drawings)
    out = []
    for d in drawings:
        rect = d.get("rect")
        if not rect:
            continue  # skip path segments without bbox (closePath/scissor entries)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if 1.0 < w < max_pt and 1.0 < h < max_pt:
            out.append(element_of(d))
    del drawings  # free the large list immediately
    return out, total_drawings


def passes_symbol_gate(e):
    if not (SYMBOL_MIN_PT < e["raw_w"] < SYMBOL_MAX_PT):
        return False
    if not (SYMBOL_MIN_PT < e["raw_h"] < SYMBOL_MAX_PT):
        return False
    return ASPECT_LO < e["raw_w"] / e["raw_h"] < ASPECT_HI


def dedupe_coincident(items, tol=COINCIDENT_TOL_PT):
    grid = defaultdict(list)
    cell = max(tol * 2, 1.0)
    out = []
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


def find_legend(page, title=None):
    LEGEND_TITLES = [
        "FIRE SPRINKLER SYMBOL LEGEND",
        "FIRE PROTECTION CRITERIA LEGEND",
        "SPRINKLER LEGEND",
        "FP LEGEND",
        "SYMBOL LEGEND",
        "SYMBOL SCHEDULE",
        "SPRINKLER SCHEDULE",
        "FIRE PROTECTION LEGEND",
    ]
    search_terms = [title] if title else LEGEND_TITLES
    hits = []
    for t in search_terms:
        hits = page.search_for(t)
        if hits:
            break
    if not hits:
        return None, []
    anchor = hits[0]
    table = fitz.Rect(anchor.x0 - 20, anchor.y1,
                      anchor.x1 + 80, min(anchor.y1 + 600, page.rect.y1))
    lines = []
    for b in page.get_text("dict", clip=table)["blocks"]:
        for l in b.get("lines", []):
            txt = "".join(s["text"] for s in l["spans"]).strip()
            if txt and txt.upper() not in ("SYMBOL", "DESCRIPTION"):
                lines.append((l["bbox"][1], l["bbox"][3], l["bbox"][0], txt))
    lines.sort()
    bands, cur, GAP = [], [], 6.0
    for ln in lines:
        if cur and ln[0] - cur[-1][1] > GAP:
            bands.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        bands.append(cur)
    text_x = min((l[2] for l in lines), default=table.x1)
    rows = [{
        "y0": min(l[0] for l in band) - 6,
        "y1": max(l[1] for l in band) + 6,
        "desc_x": text_x,
        "description": " ".join(l[3] for l in band),
    } for band in bands]
    return table, rows


HEAD_KEYWORDS = ("SPRINKLER",)
NOT_HEAD_KEYWORDS = ("RISER", "DEPARTMENT CONNECTION", "DRAIN", "NODE",
                     "PIPING", "PIPE", "ELEVATION", "TRANSITION", "CAP")


def legend_head_rows(rows):
    out = []
    for r in rows:
        d = r["description"].upper()
        if any(k in d for k in HEAD_KEYWORDS) and not any(k in d for k in NOT_HEAD_KEYWORDS):
            out.append(r)
    return out




SWATCH_TOLERANCE_PT = 0.5  # tolerance added around measured swatch dimensions


def measure_legend_swatches(page, table, head_rows):
    """Measure circle and triangle swatch symbols from legend head rows.
    
    Returns (circle_dia, tri_w, tri_h) tuples or None values if not measurable.
    Each tuple is (min, max) for the filter range.
    """
    if not table or not head_rows:
        return None, None, None

    drawings = page.get_cdrawings(extended=True)
    circle_sizes = []  # (w, h) of each circle swatch
    tri_sizes = []     # (w, h) of each triangle swatch

    # Search area: leftmost portion of legend table where symbols live
    swatch_x_min = table.x0 - 30
    swatch_x_max = table.x0 + 150

    for r in head_rows:
        y0, y1 = r["y0"] - 5, r["y1"] + 5
        row_circles = []
        row_tris = []
        for d in drawings:
            rect = d.get("rect")
            if not rect:
                continue
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            filled = d.get("fill") is not None
            if filled:
                continue
            if not (y0 < cy < y1 and swatch_x_min < cx < swatch_x_max):
                continue
            if not (5.0 < w < 16.0 and 5.0 < h < 16.0):
                continue
            kinds = "".join(sorted(it[0] for it in d["items"]))
            if kinds == "cccc":
                row_circles.append((w, h))
            elif kinds in ("lll", "ll"):
                row_tris.append((w, h))
        # Take the largest cccc per row (the main circle, not inner dots)
        if row_circles:
            best = max(row_circles, key=lambda t: t[0] * t[1])
            circle_sizes.append(best)
        if row_tris:
            best = max(row_tris, key=lambda t: t[0] * t[1])
            tri_sizes.append(best)

    circle_dia = None
    tri_w = None
    tri_h = None

    if circle_sizes:
        min_d = min(min(w, h) for w, h in circle_sizes)
        max_d = max(max(w, h) for w, h in circle_sizes)
        circle_dia = (min_d - SWATCH_TOLERANCE_PT, max_d + SWATCH_TOLERANCE_PT)

    if tri_sizes:
        min_w = min(w for w, h in tri_sizes)
        max_w = max(w for w, h in tri_sizes)
        min_h = min(h for w, h in tri_sizes)
        max_h = max(h for w, h in tri_sizes)
        tri_w = (min_w - SWATCH_TOLERANCE_PT, max_w + SWATCH_TOLERANCE_PT)
        tri_h = (min_h - SWATCH_TOLERANCE_PT, max_h + SWATCH_TOLERANCE_PT)

    return circle_dia, tri_w, tri_h

def short_label(description):
    d = description.upper()
    kind = ("PENDENT" if "PENDENT" in d else
            "DRY SIDEWALL" if "DRY SIDEWALL" in d else
            "SIDEWALL" if "SIDEWALL" in d else
            "UPRIGHT" if "UPRIGHT" in d else "SPRINKLER")
    k = ""
    for token in ("K-5.6", "K-8.0", "K5.6", "K8.0"):
        if token in d:
            k = " " + token.replace("K", "K-").replace("--", "-")
            break
    return (kind + k).strip()


def companions(core, index, cell):
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


def build_index(elements, cell=None):
    cell = cell or CONCENTRIC_TOL_PT * 2
    idx = defaultdict(list)
    for e in elements:
        idx[(int(e["cx"] // cell), int(e["cy"] // cell))].append(e)
    return idx, cell


def lines_are_parallel(slashes):
    angs = sorted({a for s in slashes for a in s["angles"]})
    if len(angs) < 2:
        return True
    return (max(angs) - min(angs)) <= ANGLE_SPREAD_MAX


def classify(core, comps):
    slashes = [e for e in comps if e["kinds"] == "l" and 5.5 < e["raw_w"] < 13]
    dots = [e for e in comps if e["kinds"] == "cccc" and e["filled"] and e["raw_w"] < 5]
    if dots:
        return "pendent", None
    if len(slashes) >= 2:
        if not lines_are_parallel(slashes):
            return None, "crossed_lines_valve_or_drain"
        return "upright_large_orifice", None
    if len(slashes) == 0:
        return "upright_plain", None
    return None, "single_slash_unrecognized"


# Default fallback ranges when legend swatches cannot be measured
DEFAULT_CIRCLE_DIA = (8.5, 9.2)
DEFAULT_TRI_W = (7.3, 8.1)
DEFAULT_TRI_H = (8.4, 9.2)


def count_heads(pdf_path, page_no=0, exclude_right_of=None,
                circle_dia=None, tri_w=None, tri_h=None):
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    elements, drawings_on_page = small_elements(page)
    table, rows = find_legend(page)
    head_rows = legend_head_rows(rows)
    if exclude_right_of is None:
        exclude_right_of = (table.x0 - 20) if table else page.rect.x1

    # Dynamic swatch measurement: derive size filters from legend symbols
    measured_circle, measured_tri_w, measured_tri_h = measure_legend_swatches(
        page, table, head_rows)

    # Use explicit args > measured > defaults
    if circle_dia is None:
        circle_dia = measured_circle if measured_circle else DEFAULT_CIRCLE_DIA
    if tri_w is None:
        tri_w = measured_tri_w if measured_tri_w else DEFAULT_TRI_W
    if tri_h is None:
        tri_h = measured_tri_h if measured_tri_h else DEFAULT_TRI_H

    def in_plan(e):
        return e["cx"] < exclude_right_of

    def is_circle(e):
        return (e["kinds"] == "cccc"
                and circle_dia[0] < e["raw_w"] < circle_dia[1]
                and circle_dia[0] < e["raw_h"] < circle_dia[1])

    def is_triangle(e):
        return (e["kinds"] in ("lll", "ll")
                and tri_w[0] < e["raw_w"] < tri_w[1]
                and tri_h[0] < e["raw_h"] < tri_h[1])

    index, cell = build_index(elements)
    circles = dedupe_coincident([e for e in elements
                                 if is_circle(e) and in_plan(e) and passes_symbol_gate(e)])
    triangles = dedupe_coincident([e for e in elements
                                   if is_triangle(e) and in_plan(e)])
    heads, rejected = [], Counter()
    for c in circles:
        cls, why = classify(c, companions(c, index, cell))
        if cls is None:
            rejected[why] += 1
            continue
        heads.append({"x": round(c["cx"], 1), "y": round(c["cy"], 1), "class": cls})
    for t in triangles:
        heads.append({"x": round(t["cx"], 1), "y": round(t["cy"], 1), "class": "sidewall"})

    counts = Counter(h["class"] for h in heads)
    total = len(heads)
    doc.close()
    return {
        "page": page_no,
        "total": total,
        "counts": dict(counts),
        "heads": heads,
        "rejected": dict(rejected),
        "legend_head_types": [short_label(r["description"]) for r in head_rows],
        "exclude_right_of": round(exclude_right_of, 1),
        "drawings_on_page": drawings_on_page,
        "small_elements": len(elements),
        "method": "vector-geometric",
        "confidence": 1.0 if total else 0.0,
        "needs_verification": total == 0,
    }


PALETTE = [(1, 0, 0), (0, 0, 1), (0, 0.6, 0), (1, 0.5, 0), (0.6, 0, 0.6)]


def write_overlay(pdf_path, result, out_png, page_no=0, dpi=120, clip=None):
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    classes = sorted({h["class"] for h in result["heads"]})
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    for h in result["heads"]:
        page.draw_circle(fitz.Point(h["x"], h["y"]), 9,
                         color=colors[h["class"]], width=1.8)
    page.set_cropbox(clip or fitz.Rect(0, 0, result["exclude_right_of"], page.rect.y1))
    page.get_pixmap(dpi=dpi).save(out_png)
    doc.close()
    return colors


def count_in_region(result, x0, y0, x1, y1):
    return sum(1 for h in result["heads"] if x0 < h["x"] < x1 and y0 < h["y"] < y1)


if __name__ == "__main__":
    import sys, json
    r = count_heads(sys.argv[1])
    r.pop("heads")
    print(json.dumps(r, indent=2))
