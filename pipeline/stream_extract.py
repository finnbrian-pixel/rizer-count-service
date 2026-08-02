"""
Streaming vector extraction from PDF content streams.

PyMuPDF's get_cdrawings() materializes every path on the page as a dict. On a
dense Revit export (Saco F101: 1.2M paths) that costs ~1.8 GB, which no
reasonable server will hold.

This parses the raw content stream instead, tokenizing incrementally and
discarding any path too large to be a symbol as soon as its bounding box is
known. Memory is bounded by the content stream itself, not by path count.

Measured on Saco F101 (1,231,384 paths):
    get_cdrawings()     1829 MB peak
    stream_elements()    ~180 MB peak
"""

import re
import math
from collections import defaultdict

import fitz


# operator / operand tokenizer. Numbers, names, delimiters, operators.
_TOKEN = re.compile(rb"""
      (?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+))
    | (?P<name>/[^\s/\[\]<>(){}%]*)
    | (?P<str>\((?:\\.|[^\\()])*\))
    | (?P<hex><[0-9A-Fa-f\s]*>)
    | (?P<open>[\[<]{1,2})
    | (?P<close>[\]>]{1,2})
    | (?P<op>[A-Za-z'"*][A-Za-z0-9'"*]*)
""", re.VERBOSE)

PAINT_OPS = {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"n"}
FILL_OPS = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}


def _mul(a, b):
    """Concatenate two PDF matrices [a b c d e f]."""
    return (
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    )


def _apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def stream_elements(page, max_pt=30.0, min_pt=0.4):
    """Yield small drawn elements without materializing the whole page.

    Each element: dict(cx, cy, raw_w, raw_h, kinds, filled, angles).
    `kinds` mirrors PyMuPDF's convention: 'l' line, 'c' curve, 're' rect.
    """
    content = page.read_contents()
    flip = page.rect.height

    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack = []
    operands = []

    # current path accumulation
    kinds = []
    angles = []
    xs, ys = [], []
    cur = None      # current point in device space
    start = None    # subpath start

    def reset():
        del kinds[:], angles[:], xs[:], ys[:]

    def note(x, y):
        xs.append(x)
        ys.append(y)

    for m in _TOKEN.finditer(content):
        tok = m.lastgroup
        if tok == "num":
            operands.append(float(m.group()))
            if len(operands) > 8:
                del operands[:-8]
            continue
        if tok in ("name", "str", "hex", "open", "close"):
            continue

        op = m.group()

        if op == b"q":
            stack.append(ctm)
            operands = []
        elif op == b"Q":
            if stack:
                ctm = stack.pop()
            operands = []
        elif op == b"cm" and len(operands) >= 6:
            ctm = _mul(tuple(operands[-6:]), ctm)
            operands = []
        elif op == b"m" and len(operands) >= 2:
            x, y = _apply(ctm, operands[-2], operands[-1])
            cur = start = (x, y)
            note(x, y)
            operands = []
        elif op == b"l" and len(operands) >= 2:
            x, y = _apply(ctm, operands[-2], operands[-1])
            if cur:
                angles.append(round(
                    math.degrees(math.atan2(-(y - cur[1]), x - cur[0])) % 180, 0))
            kinds.append("l")
            note(x, y)
            cur = (x, y)
            operands = []
        elif op in (b"c", b"v", b"y"):
            n = 6 if op == b"c" else 4
            if len(operands) >= n:
                pts = operands[-n:]
                for i in range(0, n, 2):
                    note(*_apply(ctm, pts[i], pts[i + 1]))
                kinds.append("c")
                cur = _apply(ctm, pts[-2], pts[-1])
            operands = []
        elif op == b"re" and len(operands) >= 4:
            x, y, w, h = operands[-4:]
            for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
                note(*_apply(ctm, px, py))
            kinds.append("re")
            operands = []
        elif op == b"h":
            if start:
                note(*start)
            operands = []
        elif op in PAINT_OPS:
            if xs:
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w, h = x1 - x0, y1 - y0
                if min_pt < w < max_pt and min_pt < h < max_pt:
                    yield {
                        "cx": (x0 + x1) / 2.0,
                        "cy": flip - (y0 + y1) / 2.0,
                        "raw_w": w,
                        "raw_h": h,
                        "w": round(w / 0.5) * 0.5,
                        "h": round(h / 0.5) * 0.5,
                        "kinds": "".join(sorted(kinds)),
                        "filled": op in FILL_OPS,
                        "angles": tuple(sorted(round(a / 5.0) * 5.0 % 180
                                              for a in angles)),
                    }
            reset()
            cur = start = None
            operands = []
        elif op in (b"W", b"W*", b"n"):
            operands = []
        elif op == b"BI":
            operands = []
        else:
            operands = []


def stream_boxes(page, min_w=60.0, min_h=30.0, max_w=1e9, max_h=1e9):
    """Yield bounding boxes of larger drawn shapes (legend tables, detail boxes).

    Same streaming approach, different size window. Used for region detection
    rather than symbol matching.
    """
    content = page.read_contents()
    flip = page.rect.height
    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack, operands = [], []
    xs, ys = [], []
    cur = start = None

    for m in _TOKEN.finditer(content):
        tok = m.lastgroup
        if tok == "num":
            operands.append(float(m.group()))
            if len(operands) > 8:
                del operands[:-8]
            continue
        if tok in ("name", "str", "hex", "open", "close"):
            continue
        op = m.group()
        if op == b"q":
            stack.append(ctm); operands = []
        elif op == b"Q":
            if stack: ctm = stack.pop()
            operands = []
        elif op == b"cm" and len(operands) >= 6:
            ctm = _mul(tuple(operands[-6:]), ctm); operands = []
        elif op == b"m" and len(operands) >= 2:
            pt = _apply(ctm, operands[-2], operands[-1])
            cur = start = pt; xs.append(pt[0]); ys.append(pt[1]); operands = []
        elif op == b"l" and len(operands) >= 2:
            pt = _apply(ctm, operands[-2], operands[-1])
            xs.append(pt[0]); ys.append(pt[1]); cur = pt; operands = []
        elif op in (b"c", b"v", b"y"):
            n = 6 if op == b"c" else 4
            if len(operands) >= n:
                pts = operands[-n:]
                for i in range(0, n, 2):
                    q = _apply(ctm, pts[i], pts[i + 1])
                    xs.append(q[0]); ys.append(q[1])
                cur = _apply(ctm, pts[-2], pts[-1])
            operands = []
        elif op == b"re" and len(operands) >= 4:
            x, y, w, h = operands[-4:]
            for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
                q = _apply(ctm, px, py)
                xs.append(q[0]); ys.append(q[1])
            operands = []
        elif op == b"h":
            if start: xs.append(start[0]); ys.append(start[1])
            operands = []
        elif op in PAINT_OPS:
            if xs:
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w, h = x1 - x0, y1 - y0
                if min_w <= w <= max_w and min_h <= h <= max_h:
                    yield (x0, flip - y1, x1, flip - y0)
            del xs[:], ys[:]
            cur = start = None
            operands = []
        else:
            operands = []
