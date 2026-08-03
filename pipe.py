"""
Pipe takeoff from vector fire protection drawings.

Linear feet by nominal size, measured from the drawing's own geometry.

The method mirrors the head counter: nothing about this sheet is hardcoded.
Sprinkler pipe is drawn at a distinct stroke weight, but that weight differs by
engineer and CAD package — so it is *learned* from the drawing by looking at
which stroke width the segments touching sprinkler heads actually use. Sizes
come from the drawing's own size labels ("2\"", "1-1/4\"").

Verified on ITD FM32361 FS-01.
"""

import math
import re
from collections import Counter, defaultdict

import fitz

import stream_extract as se


MIN_SEG_PT = 2.0          # below this, a segment is a symbol stroke not pipe
HEAD_TOL_PT = 8.0         # endpoint-to-head distance that counts as "connected"
LABEL_MAX_PT = 30.0       # a size label this far from a run applies to it
MIN_VOTE_LEN_PT = 20.0    # only real runs vote on the pipe stroke weight
SYMBOL_GAP_PT = 14.0      # CAD breaks pipe at each head; bridge gaps this size
WIDTH_TOL = 0.05

SIZE_RE = re.compile(r'^\d+(?:-\d/\d)?"$|^\d/\d"$')


# ───────────────────────────────────────────────────────── segment extraction ─

def extract_segments(page, min_len=MIN_SEG_PT, x_limit=None):
    """Stream every drawn line segment with its stroke width, in PDF points.

    Same streaming approach as the head counter — never materializes the page.
    """
    content = page.read_contents()
    flip = page.rect.height
    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack, widths, operands = [], [], []
    lw = 1.0
    cur = None
    out = []

    for m in se._TOKEN.finditer(content):
        kind = m.lastgroup
        if kind == "num":
            operands.append(float(m.group()))
            if len(operands) > 8:
                del operands[:-8]
            continue
        if kind in ("name", "str", "hex", "open", "close"):
            continue

        op = m.group()
        if op == b"q":
            stack.append(ctm); widths.append(lw); operands = []
        elif op == b"Q":
            if stack: ctm = stack.pop()
            if widths: lw = widths.pop()
            operands = []
        elif op == b"cm" and len(operands) >= 6:
            ctm = se._mul(tuple(operands[-6:]), ctm); operands = []
        elif op == b"w" and operands:
            lw = operands[-1]; operands = []
        elif op == b"m" and len(operands) >= 2:
            cur = se._apply(ctm, operands[-2], operands[-1]); operands = []
        elif op == b"l" and len(operands) >= 2:
            p = se._apply(ctm, operands[-2], operands[-1])
            if cur:
                L = math.hypot(p[0] - cur[0], p[1] - cur[1])
                if L >= min_len:
                    sx = math.hypot(ctm[0], ctm[1])
                    x0, y0, x1, y1 = cur[0], flip - cur[1], p[0], flip - p[1]
                    if x_limit is None or max(x0, x1) < x_limit:
                        out.append({
                            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                            "len": L, "width": round(lw * sx, 2),
                        })
            cur = p
            operands = []
        elif op in (b"c", b"v", b"y"):
            n = 6 if op == b"c" else 4
            if len(operands) >= n:
                cur = se._apply(ctm, operands[-2], operands[-1])
            operands = []
        else:
            operands = []
    return out


# ─────────────────────────────────────────────────────────── width learning ──

def learn_pipe_width(segments, heads, tol=HEAD_TOL_PT, min_vote_len=MIN_VOTE_LEN_PT):
    """Which stroke weight is pipe? Ask the heads.

    Sprinklers sit on pipe, so the stroke weight used by segments ending at a
    head is the pipe weight for this drawing. Learned per sheet, never assumed.

    Only segments long enough to be a pipe run get a vote. A sprinkler symbol is
    itself drawn with short strokes that touch its own centre, and those strokes
    outnumber the real pipe if allowed to vote — which picks the wrong weight
    and roughly quadruples the takeoff.
    """
    votes = Counter()
    for s in segments:
        if s["len"] >= min_vote_len and _endpoint_near_head(s, heads, tol):
            votes[s["width"]] += 1
    if not votes:
        return None, votes
    return votes.most_common(1)[0][0], votes


def _endpoint_near_head(s, heads, tol):
    for hx, hy in heads:
        if abs(s["x0"] - hx) < tol and abs(s["y0"] - hy) < tol:
            return True
        if abs(s["x1"] - hx) < tol and abs(s["y1"] - hy) < tol:
            return True
    return False


# ─────────────────────────────────────────────────────────── size labelling ──

def size_labels(page, x_limit=None):
    """Pipe size callouts on the sheet, e.g. 2" / 1-1/4" / 3/4"."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            txt = "".join(sp["text"] for sp in line["spans"]).strip()
            if not SIZE_RE.match(txt):
                continue
            x0, y0, x1, y1 = line["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if x_limit is None or cx < x_limit:
                out.append({"cx": cx, "cy": cy, "size": txt})
    return out


def _point_to_segment(px, py, s):
    dx, dy = s["x1"] - s["x0"], s["y1"] - s["y0"]
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - s["x0"], py - s["y0"])
    t = max(0.0, min(1.0, ((px - s["x0"]) * dx + (py - s["y0"]) * dy) / L2))
    return math.hypot(px - (s["x0"] + t * dx), py - (s["y0"] + t * dy))


def bridge_symbol_gaps(segments, axis_tol=1.5, max_gap=SYMBOL_GAP_PT):
    """Rejoin runs that CAD broke at each sprinkler symbol.

    A branch line is physically continuous, but the drawing interrupts it where
    each head symbol sits — roughly a foot of gap per head. Counting only the
    drawn segments understates the installed pipe (7% on ITD FS-01, 133 ft).

    Collinear segments separated by less than a symbol's width are merged; a
    genuine gap between separate runs is far wider and survives.
    """
    verts, horiz, diagonal = defaultdict(list), defaultdict(list), []
    for s in segments:
        if abs(s["x0"] - s["x1"]) < axis_tol:
            key = round((s["x0"] + s["x1"]) / 2 / axis_tol) * axis_tol
            verts[key].append((min(s["y0"], s["y1"]), max(s["y0"], s["y1"]), s))
        elif abs(s["y0"] - s["y1"]) < axis_tol:
            key = round((s["y0"] + s["y1"]) / 2 / axis_tol) * axis_tol
            horiz[key].append((min(s["x0"], s["x1"]), max(s["x0"], s["x1"]), s))
        else:
            diagonal.append(s)

    out = list(diagonal)
    for group, vertical in ((verts, True), (horiz, False)):
        for key, items in group.items():
            items.sort()
            lo, hi, ref = items[0]
            for a, b, seg in items[1:]:
                if a - hi <= max_gap:
                    hi = max(hi, b)
                else:
                    out.append(_run(key, lo, hi, vertical, ref))
                    lo, hi, ref = a, b, seg
            out.append(_run(key, lo, hi, vertical, ref))
    return out


def _run(key, lo, hi, vertical, ref):
    if vertical:
        return {"x0": key, "y0": lo, "x1": key, "y1": hi,
                "len": hi - lo, "width": ref["width"]}
    return {"x0": lo, "y0": key, "x1": hi, "y1": key,
            "len": hi - lo, "width": ref["width"]}


def assign_sizes(pipe, labels, max_dist=LABEL_MAX_PT):
    """Attach the nearest size callout to each pipe segment."""
    for s in pipe:
        best, bd = None, float("inf")
        for lab in labels:
            d = _point_to_segment(lab["cx"], lab["cy"], s)
            if d < bd:
                bd, best = d, lab["size"]
        s["size"] = best if bd <= max_dist else None
        s["label_dist"] = round(bd, 1)
    return pipe


# ──────────────────────────────────────────────────────────────── the takeoff ─

def pipe_takeoff(pdf_path, page_no=0, heads=None, ft_per_pt=None, exclude_right_of=None):
    """Linear feet of sprinkler pipe by nominal size.

    heads: [(x, y)] in PDF points. If omitted, taken from the head counter.
    """
    import hc2
    import physics

    doc = fitz.open(pdf_path)
    page = doc[page_no]

    if heads is None or ft_per_pt is None or exclude_right_of is None:
        res = hc2.count_page(pdf_path, page_no)
        if heads is None:
            heads = [(h["x"], h["y"]) for h in res["heads"]]
        if exclude_right_of is None:
            zones = res.get("excluded_zones") or []
            exclude_right_of = min((z[0] for z in zones), default=page.rect.x1)
        if ft_per_pt is None:
            ft_per_pt = physics.parse_scale(page)

    segments = extract_segments(page, x_limit=exclude_right_of)
    width, votes = learn_pipe_width(segments, heads)

    result = {
        "page": page_no,
        "segments_considered": len(segments),
        "pipe_stroke_width": width,
        "width_votes": dict(votes.most_common(4)),
        "scale_ft_per_pt": ft_per_pt,
        "heads": len(heads),
    }

    if width is None or ft_per_pt is None:
        result.update({
            "total_ft": 0.0, "by_size": {}, "unassigned_ft": 0.0,
            "heads_connected": 0, "confidence": 0.0, "needs_verification": True,
            "reason": "could not learn pipe stroke width" if width is None
                      else "sheet scale not found",
        })
        doc.close()
        return result

    raw = [s for s in segments if abs(s["width"] - width) < WIDTH_TOL]
    drawn_ft = sum(s["len"] for s in raw) * ft_per_pt
    pipe = bridge_symbol_gaps(raw)
    pipe = assign_sizes(pipe, size_labels(page, x_limit=exclude_right_of))

    by_size, unassigned = defaultdict(float), 0.0
    for s in pipe:
        if s["size"]:
            by_size[s["size"]] += s["len"] * ft_per_pt
        else:
            unassigned += s["len"] * ft_per_pt

    connected = sum(1 for hx, hy in heads
                    if any(_endpoint_near_head(s, [(hx, hy)], HEAD_TOL_PT) for s in raw))
    coverage = connected / len(heads) if heads else 0.0
    total = sum(by_size.values()) + unassigned

    # Confidence is grounded in something checkable: what share of the heads the
    # detected network actually reaches. A network that misses heads is missing
    # pipe, whatever its total says.
    result.update({
        "total_ft": round(total, 1),
        "by_size": {k: round(v, 1) for k, v in
                    sorted(by_size.items(), key=lambda kv: -kv[1])},
        "unassigned_ft": round(unassigned, 1),
        "pipe_segments": len(pipe),
        "drawn_ft": round(drawn_ft, 1),
        "bridged_ft": round(total - drawn_ft, 1),
        "heads_connected": connected,
        "head_coverage": round(coverage, 3),
        "confidence": round(min(1.0, coverage), 3),
        "needs_verification": coverage < 0.95 or unassigned > total * 0.15,
        "flags": _flags(coverage, unassigned, total),
    })
    doc.close()
    return result


def _flags(coverage, unassigned, total):
    out = []
    if coverage < 0.95:
        out.append(f"{round((1-coverage)*100)}% of heads have no pipe connection — "
                   f"runs are likely missing from the total")
    if total and unassigned > total * 0.15:
        out.append(f"{round(unassigned)} ft could not be matched to a size callout")
    if not out:
        out.append("every head reaches the detected pipe network")
    return out


if __name__ == "__main__":
    import sys, json
    print(json.dumps(pipe_takeoff(sys.argv[1]), indent=2))
