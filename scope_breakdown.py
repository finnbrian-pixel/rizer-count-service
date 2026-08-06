"""
scope_breakdown.py — RIZER Construction Intelligence  (v2 — general architecture)
Fire Sprinkler (Division 21) scope extraction from commercial bid packages,
generalized across GCs / architects / spec writers.

WHY THIS VERSION EXISTS
------------------------
v1 worked perfectly on the VCPS package (12 sheets, 19 scope items, 7 Div 21
sections, 19 design criteria, 7 interfaces, 5 RFIs) and returned almost total
zeros on a second, unrelated GC's package (Fukui Architects / HACP Murray
Towers) — not because it crashed, but because every "meaning" extractor was
a literal string lifted from VCPS's spec writer, and every "structure"
extractor assumed VCPS's exact formatting (one-line "SECTION 21 00 10 -
TITLE" headers, "CC #21A - FIRE SUPPRESSION" contract-category numbering,
sheet titleblocks that print "FIRE PROTECTION" next to the sheet number).
None of that generalizes; a different GC/architect/spec author will never
reproduce another firm's exact phrasing.

ARCHITECTURE
------------
Two passes, cleanly separated:

  PASS 1 — STRUCTURE (deterministic, regex, but format-tolerant)
    Finds *where* things are: Division 21 section headers (any spacing/
    punctuation/line-break convention), the sheet index/drawing list page
    (present in virtually every set, regardless of GC), and scope-of-work
    header blocks (matched by a synonym family + fire/sprinkler keyword
    proximity, not one literal header string).

  PASS 2 — MEANING (semantic, LLM-backed)
    Every candidate block found in Pass 1 is handed to Claude with a forced
    JSON schema (via tool use) to classify and extract scope items, design
    criteria, cross-trade interfaces, and RFI-worthy ambiguous language.
    This is what actually generalizes: it reads for *meaning*, not string
    equality, so a different spec writer's phrasing is not a blind spot.

  COVERAGE TRACKING
    Pass 1's candidate-block counts are reported alongside Pass 2's
    extraction counts. If Pass 1 finds 14 Division 21 paragraphs and Pass 2
    extracts 0 design criteria from them, that is now visible as "14 found /
    0 extracted" instead of silently reporting `design_criteria: []` with no
    way to tell "nothing there" apart from "extractor missed it."

  GRACEFUL DEGRADATION
    If ANTHROPIC_API_KEY isn't set, Pass 2 is skipped and the report is
    flagged `llm_available: False` with a loud Flag — the report still
    contains everything Pass 1 found (sheets, section locations, candidate
    block count) but will not silently claim zero scope items when it
    simply never tried.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Iterable

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Depth(str, Enum):
    FULL = "full"
    FLAG = "flag"
    OFF = "off"


@dataclass
class ContractorProfile:
    company: str = "Unknown Contractor"
    trades: dict[str, Depth] = field(default_factory=lambda: {
        "fire_sprinkler": Depth.FULL,
        "fire_alarm": Depth.FLAG,
        "fire_pump": Depth.FLAG,
        "underground": Depth.FLAG,
    })


class Confidence(str, Enum):
    HIGH = "high"       # verbatim quote in the LLM's extraction
    MEDIUM = "medium"   # paraphrased / inferred from context
    LOW = "low"         # weakly supported, needs human confirmation


class Severity(str, Enum):
    INFO = "info"
    COORDINATE = "coordinate"
    RFI = "rfi"


_log = logging.getLogger("scope_breakdown")

LLM_MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# Evidence + findings  (unchanged shape — anything downstream that already
# consumes ScopeReport keeps working)
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    document: str
    page: int
    excerpt: str
    verbatim: bool = False
    """True only when `excerpt` was found character-for-character in the source
    page text. When False the excerpt is the model's paraphrase and must not be
    quoted back to a GC or owner as spec language."""

    def cite(self) -> str:
        return f"{self.document}, p.{self.page}"


@dataclass
class ScopeItem:
    label: str
    detail: str
    confidence: Confidence
    evidence: Evidence
    cost_flag: bool = False


@dataclass
class SheetRef:
    number: str
    title: str
    page: int
    level: str | None = None


@dataclass
class SpecRef:
    section: str
    title: str
    division: str
    note: str
    evidence: Evidence


@dataclass
class Interface:
    counterparty: str
    description: str
    direction: str
    evidence: Evidence


@dataclass
class Flag:
    title: str
    detail: str
    severity: Severity
    evidence: Evidence
    suggested_rfi: str | None = None


@dataclass
class DesignCriterion:
    category: str
    requirement: str
    evidence: Evidence


@dataclass
class Coverage:
    """Pass-1 candidate counts vs Pass-2 extraction counts, per category.
    Lets a zero downstream be told apart from 'nothing there' vs 'missed it'."""
    div21_blocks_found: int = 0
    scope_blocks_found: int = 0
    sheet_index_pages_found: int = 0
    titleblock_pages_scanned: int = 0
    titleblock_pages_parsed: int = 0
    llm_calls_made: int = 0
    llm_calls_failed: int = 0
    llm_last_error: str | None = None
    quotes_verified: int = 0
    quotes_unverified: int = 0


@dataclass
class ScopeReport:
    project: str
    contractor: str
    contract_category: str | None
    documents: list[dict]
    sheets: list[SheetRef]
    scope_items: list[ScopeItem]
    spec_refs: list[SpecRef]
    interfaces: list[Interface]
    flags: list[Flag]
    div21_sections: list[SpecRef] = field(default_factory=list)
    design_criteria: list[DesignCriterion] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    llm_available: bool = True

    def to_json(self, **kw) -> str:
        return json.dumps(asdict(self), indent=2, default=str, **kw)


# ---------------------------------------------------------------------------
# Document ingestion (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class Document:
    path: Path
    label: str
    pages: list[str]

    @classmethod
    def load(cls, path: str | Path, label: str | None = None) -> "Document":
        path = Path(path)
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, errors="ignore",
        )
        if out.returncode != 0:
            raise RuntimeError(f"pdftotext failed on {path.name}")
        pages = out.stdout.split("\f")
        return cls(path=path, label=label or path.stem, pages=pages)

    def page_text(self, page: int) -> str:
        return self.pages[page - 1] if 0 < page <= len(self.pages) else ""

    def search(self, pattern: str, flags=re.I) -> Iterable[tuple[int, re.Match]]:
        rx = re.compile(pattern, flags)
        for i, text in enumerate(self.pages, start=1):
            for m in rx.finditer(text):
                yield i, m

    def excerpt(self, page: int, match: re.Match, width: int = 200) -> str:
        text = self.page_text(page)
        s = max(0, match.start() - width // 3)
        e = min(len(text), match.end() + width)
        return re.sub(r"\s+", " ", text[s:e]).strip()


# ---------------------------------------------------------------------------
# PASS 1 — STRUCTURE (format-tolerant, no GC-specific literals)
# ---------------------------------------------------------------------------

# Division N section header. Tolerates:
#   - dash / em-dash / en-dash / colon / nothing as separator
#   - title on the same line OR the next non-blank line
#   - variable internal spacing ("21 00 10", "210010", "21-00-10")
DIVISION_HEADER_RX = (
    r"SECTION\s+(\d{2})\D{0,2}(\d{2})\D{0,2}(\d{2})"
    r"[ \t]*[-–—:]?[ \t]*\n*[ \t]*"
    r"([A-Z][A-Z0-9 ,&/'()\-]{4,90})"
)


def extract_division_sections(docs: list["Document"], division: str = "21") -> list[SpecRef]:
    """Locate Division N sections regardless of header punctuation/line-break style."""
    found: dict[str, SpecRef] = {}
    rx = re.compile(DIVISION_HEADER_RX)
    for doc in docs:
        for page, text in enumerate(doc.pages, start=1):
            for m in rx.finditer(text):
                if m.group(1) != division:
                    continue
                section = f"{m.group(1)} {m.group(2)}{m.group(3)}"
                title = re.sub(r"\s+", " ", m.group(4)).strip(" -:")
                if len(title) < 4 or title.isupper() is False and title[0].islower():
                    continue
                title = title.title()
                found.setdefault(section, SpecRef(
                    section=section, title=title,
                    division=f"Division {division}",
                    note="Section header located by generalized division-header pattern.",
                    evidence=Evidence(doc.label, page, f"Section {section} — {title}"),
                ))
    return sorted(found.values(), key=lambda r: r.section)


def extract_division_blocks(docs: list["Document"], division: str = "21") -> list[tuple[Document, int, str]]:
    """Return the actual text block (this page + next) for every located
    Division N section — these are the candidate blocks fed to the LLM pass."""
    sections = extract_division_sections(docs, division)
    blocks = []
    doc_by_label = {d.label: d for d in docs}
    for s in sections:
        doc = doc_by_label[s.evidence.document]
        page = s.evidence.page
        text = doc.page_text(page) + "\n" + doc.page_text(page + 1)
        blocks.append((doc, page, text))
    return blocks


# Sheet index / drawing list — present in virtually every set regardless of GC.
SHEET_INDEX_HEADER_RX = re.compile(
    r"(SHEET\s+INDEX|DRAWING\s+INDEX|LIST\s+OF\s+DRAWINGS|INDEX\s+OF\s+DRAWINGS|SHEET\s+LIST)",
    re.I,
)
# A sheet-index row: sheet number, whitespace/dots, title.
SHEET_ROW_RX = re.compile(
    r"^\s*([A-Z]{1,3}\-?\d{1,4}(?:\.\d+)?[A-Z]?)\s{2,}\.*\s*([A-Za-z][A-Za-z0-9 ,/&'\"()\-]{3,80})\s*$"
)
FIRE_DISCIPLINE_PREFIXES = ("FP", "F", "FS")


TB_DATE_RX = re.compile(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}")
TB_SHEET_CODE_RX = re.compile(r"^[A-Z]{1,4}[-.]?\d{1,4}(\.\d+)?[A-Z]?$")
TB_INT_RX = re.compile(r"^\d{1,4}$")
TB_LABEL_TERMS = ["sheet no", "drawing no", "dwg no", "sheet number"]

# Discipline prefixes that mean "fire protection scope" — sprinkler drawings
# are typically FP-prefixed; fire alarm (FA) is a related but distinct trade,
# kept separate here since it's commonly a different subcontractor's scope.
FIRE_SUPPRESSION_PREFIXES = ("FP", "FS")
FIRE_ALARM_PREFIXES = ("FA",)


def _titleblock_clip_text(page) -> str | None:
    """Anchor on the near-universal 'Sheet No.' family label and clip a
    bounded box around it — geometry finds WHERE the titleblock is without
    needing to guess a firm-specific page fraction."""
    for term in TB_LABEL_TERMS:
        found = page.search_for(term)
        if found:
            r = found[0]
            w, h = page.rect.width, page.rect.height
            box = fitz.Rect(max(0, r.x0 - 500), max(0, r.y0 - 100),
                             min(r.x0 + 700, w), min(r.y0 + 300, h))
            return page.get_text("text", clip=box)
    return None


def _parse_titleblock(text: str | None) -> tuple[str | None, str | None]:
    """Deterministic parse of the clipped titleblock text. Handles both
    dot- and slash-separated dates (different firms/consultants on the same
    package commonly use different CAD title-block templates — an architect
    and their MEP/FP sub will often differ)."""
    if not text:
        return None, None
    tokens = [t.strip() for t in re.split(r"[|\n]", text) if t.strip()]
    date_idx = next((i for i, t in enumerate(tokens) if TB_DATE_RX.search(t)), None)
    if date_idx is None:
        return None, None
    sheet_no, start = None, None
    for j in range(date_idx + 1, min(date_idx + 4, len(tokens))):
        if TB_SHEET_CODE_RX.match(tokens[j]):
            sheet_no, start = tokens[j], j + 1
            break
    if sheet_no is None:
        return None, None
    if start < len(tokens) and TB_INT_RX.match(tokens[start]):
        start += 1
    title_tokens = []
    for t in tokens[start:start + 6]:
        if TB_INT_RX.match(t) and len(t) <= 3:
            break
        title_tokens.append(t)
    return sheet_no, " ".join(title_tokens).strip(" ,")


def extract_sheets_by_titleblock(doc: "Document", pdf_path: str) -> tuple[list[SheetRef], int, int]:
    """Fallback (and often primary) sheet discovery: scan every page's
    titleblock directly rather than relying on a compiled index page, since
    large-format multi-sheet sets frequently don't have one. Returns
    (fire-relevant sheets, pages scanned, pages successfully parsed)."""
    if fitz is None:
        return [], 0, 0
    pdf = fitz.open(pdf_path)
    sheets: dict[str, SheetRef] = {}
    parsed = 0
    for i in range(len(pdf)):
        page = pdf[i]
        w, h = page.rect.width, page.rect.height
        if max(w, h) < 900:
            continue  # not a full-size drawing sheet (likely a spec/text page)
        text = _titleblock_clip_text(page)
        num, title = _parse_titleblock(text)
        if not num:
            continue
        parsed += 1
        prefix_m = re.match(r"[A-Z]+", num)
        prefix = prefix_m.group(0) if prefix_m else ""
        is_fire = (prefix in FIRE_SUPPRESSION_PREFIXES + FIRE_ALARM_PREFIXES
                   or (title and re.search(r"FIRE\s+PROTECTION|SPRINKLER", title, re.I)))
        if is_fire:
            sheets.setdefault(num, SheetRef(number=num, title=(title or "").title(), page=i + 1))
    scanned = sum(1 for i in range(len(pdf)) if max(pdf[i].rect.width, pdf[i].rect.height) >= 900)
    return sorted(sheets.values(), key=lambda s: s.number), scanned, parsed


def extract_sheet_index(doc: "Document") -> tuple[list[SheetRef], int]:
    """Parse the drawing index page(s) rather than hunting titleblocks.
    Returns (sheets, index_pages_found)."""
    index_pages = [p for p, _ in doc.search(SHEET_INDEX_HEADER_RX.pattern)]
    sheets: list[SheetRef] = []
    for p in index_pages:
        for offset in range(0, 3):  # index often spans a couple pages
            text = doc.page_text(p + offset)
            for line in text.splitlines():
                m = SHEET_ROW_RX.match(line)
                if not m:
                    continue
                num, title = m.group(1).upper(), m.group(2).strip()
                prefix = re.match(r"[A-Z]+", num)
                prefix = prefix.group(0) if prefix else ""
                is_fire = prefix in FIRE_DISCIPLINE_PREFIXES or "FIRE" in title.upper() or "SPRINKLER" in title.upper()
                if is_fire:
                    sheets.append(SheetRef(number=num, title=title.title(), page=p + offset))
    # de-dup by sheet number
    seen: dict[str, SheetRef] = {}
    for s in sheets:
        seen.setdefault(s.number, s)
    return sorted(seen.values(), key=lambda s: s.number), len(index_pages)


# Scope-of-work section — synonym family instead of one GC's literal header,
# combined with fire/sprinkler keyword proximity so we don't grab every
# trade's scope block.
SCOPE_HEADER_RX = re.compile(
    r"(SCOPE\s+OF\s+WORK|CONTRACT\s+CATEGORY|TRADE\s+PACKAGE|BID\s+PACKAGE"
    r"|WORK\s+INCLUDED|DESCRIPTION\s+OF\s+WORK|WORK\s+SCOPE)",
    re.I,
)
FIRE_KEYWORD_RX = re.compile(r"FIRE\s+(SUPPRESSION|PROTECTION|SPRINKLER)|SPRINKLER", re.I)


def locate_scope_blocks(docs: list["Document"], window_chars: int = 4000) -> list[tuple[Document, int, str, str]]:
    """Find scope-of-work header blocks that are about fire protection, by
    proximity rather than literal header text. Returns (doc, page, header, block)."""
    out = []
    for doc in docs:
        for page, m in doc.search(SCOPE_HEADER_RX.pattern):
            text = doc.page_text(page)
            start = m.start()
            window = text[start:start + window_chars]
            if not FIRE_KEYWORD_RX.search(window[:600]):
                continue  # this scope header isn't the fire-protection one
            out.append((doc, page, m.group(0), window))
    return out


# ---------------------------------------------------------------------------
# PASS 2 — MEANING (LLM semantic extraction, forced JSON schema)
# ---------------------------------------------------------------------------

EXTRACTION_TOOL = {
    "name": "record_findings",
    "description": "Record fire-sprinkler-relevant scope items, design criteria, cross-trade interfaces, and RFI-worthy ambiguities found in this text block.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scope_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Short label, <10 words"},
                        "detail": {"type": "string", "description": "The scope item, close to verbatim"},
                        "cost_flag": {"type": "boolean", "description": "True if this item carries material/labor cost commonly missed in a sprinkler bid (firestopping, patching, permits, training, spare stock, etc.)"},
                        "quote": {"type": "string", "description": "VERBATIM span copied character-for-character from the source text that supports this item. Copy, do not paraphrase or fix typos. 10-40 words."},
                    },
                    "required": ["label", "detail", "cost_flag", "quote"],
                },
            },
            "design_criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "e.g. System type, Water supply, Head guards, Testing, AHJ approval"},
                        "requirement": {"type": "string", "description": "The technical requirement, close to verbatim"},
                        "quote": {"type": "string", "description": "VERBATIM span copied character-for-character from the source text. Copy, do not paraphrase. 10-40 words."},
                    },
                    "required": ["category", "requirement", "quote"],
                },
            },
            "interfaces": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "counterparty": {"type": "string", "description": "e.g. Electrical, Plumbing, Elevator, Fire Alarm"},
                        "description": {"type": "string"},
                        "direction": {"type": "string", "enum": ["they_provide", "we_provide", "coordinate"]},
                        "quote": {"type": "string", "description": "VERBATIM span from the source text if one supports this interface. OPTIONAL — omit it if the interface is implied rather than stated in a single sentence. Never paraphrase into this field."},
                    },
                    "required": ["counterparty", "description", "direction"],
                },
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "detail": {"type": "string"},
                        "severity": {"type": "string", "enum": ["info", "coordinate", "rfi"]},
                        "suggested_rfi": {"type": "string", "description": "Only if severity is rfi; omit otherwise"},
                        "quote": {"type": "string", "description": "VERBATIM span that triggered this flag, if one exists. OPTIONAL — many flags concern what the document FAILS to say, which has no sentence to quote. Omit rather than paraphrase."},
                    },
                    "required": ["title", "detail", "severity"],
                },
            },
        },
        "required": ["scope_items", "design_criteria", "interfaces", "flags"],
    },
}

SYSTEM_PROMPT = """You are a fire sprinkler estimator's assistant reading a construction bid
package. You are given one text block (a scope-of-work section or a Division 21
specification section). Extract ONLY what pertains to the fire sprinkler /
fire suppression trade — ignore other trades' scope even if mentioned in passing,
unless it defines a handoff/interface that affects sprinkler scope.

Rules:
- Extract meaning, not exact GC phrasing — different documents word the same
  requirement differently; treat synonyms as equivalent (e.g. "as necessary",
  "if required", "TBD", "to be determined" are all ambiguity signals worth an
  RFI flag).
- Do not invent items not supported by the text.
- The `quote` field is NOT a place to paraphrase. Copy an exact span of
  characters out of the block, including its original punctuation and any
  typos. Every quote is checked against the source text and silently
  discarded if it does not match, so a paraphrase there loses the citation.
- For scope_items and design_criteria a quote is REQUIRED — these are claims
  about what the spec says, and must be citable.
- For flags and interfaces a quote is OPTIONAL. Many of the most valuable
  flags concern what the document FAILS to say (a missing size, an unstated
  code edition, an undefined extent) and no sentence states an omission.
  NEVER suppress a flag or interface because you cannot find a quote for it —
  report it and omit the quote field.
- cost_flag = true only for items that carry real, commonly-missed cost
  (firestopping, patching, permits, escutcheons, access panels, training,
  spare stock, cleanup) — not routine installation work.
- severity=rfi for any requirement whose quantity/extent/scope boundary is
  left undefined by the text.
- If the block contains nothing relevant to fire sprinkler scope, return
  empty arrays for everything. Do not force findings.
Call record_findings with your results."""


def _get_client():
    if anthropic is None:
        return None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def llm_extract(client, block_text: str, doc_label: str, page: int, coverage: Coverage) -> dict:
    """One semantic extraction call over a candidate block. Returns dict of
    lists matching the tool schema, or empty dict on failure (never raises)."""
    coverage.llm_calls_made += 1
    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "record_findings"},
            messages=[{"role": "user", "content": block_text[:12000]}],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "record_findings":
                return block.input
        return {}
    except Exception as exc:
        # Silent failure here cost a full debugging cycle once: every call
        # 400'd and the report simply showed empty lists. Log loudly and keep
        # the first error on the coverage object so it surfaces in the report.
        coverage.llm_calls_failed += 1
        if coverage.llm_last_error is None:
            coverage.llm_last_error = f"{type(exc).__name__}: {exc}"[:400]
        _log.warning("llm_extract failed (%s p.%s): %s", doc_label, page, exc)
        return {}


_WS_RX = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    """Collapse the differences that PDF text extraction introduces but that
    don't change what the document actually says: runs of whitespace, the
    various dash and quote characters, and case."""
    s = _WS_RX.sub(" ", s)
    for a, b in (("\u2014", "-"), ("\u2013", "-"), ("\u2019", "'"),
                 ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\ufb01", "fi"), ("\ufb02", "fl"), ("\u00a0", " ")):
        s = s.replace(a, b)
    return s.strip().lower()


def _verify_quote(quote: str, source: str) -> str | None:
    """Return the quote if it genuinely appears in `source`, else None.

    The model is asked to copy verbatim, but a model can drift or hallucinate a
    plausible-sounding line. An unverified quote is worse than no quote — it
    looks like evidence. So every quote is checked against the extracted page
    text before it is allowed to be called verbatim.
    """
    if not quote or len(quote) < 12:
        return None
    if _normalize_for_match(quote) in _normalize_for_match(source):
        return quote.strip()
    return None


def _to_dataclasses(raw: dict, doc_label: str, page: int, excerpt_src: str,
                    coverage: "Coverage | None" = None) -> tuple[
        list[ScopeItem], list[DesignCriterion], list[Interface], list[Flag]]:
    items, criteria, interfaces, flags = [], [], [], []

    def ev(obj: dict, fallback_key: str) -> Evidence:
        """Prefer a verified verbatim quote; fall back to the paraphrase, but
        label it so downstream consumers know not to treat it as spec text."""
        verified = _verify_quote(obj.get("quote", ""), excerpt_src)
        if verified:
            if coverage is not None:
                coverage.quotes_verified += 1
            return Evidence(doc_label, page, verified[:300], verbatim=True)
        if coverage is not None:
            coverage.quotes_unverified += 1
        return Evidence(doc_label, page, obj.get(fallback_key, excerpt_src)[:300],
                        verbatim=False)

    for si in raw.get("scope_items", []):
        items.append(ScopeItem(
            label=si.get("label", "")[:80],
            detail=si.get("detail", ""),
            confidence=Confidence.MEDIUM,
            evidence=ev(si, "detail"),
            cost_flag=bool(si.get("cost_flag", False)),
        ))
    for dc in raw.get("design_criteria", []):
        criteria.append(DesignCriterion(
            category=dc.get("category", "General"),
            requirement=dc.get("requirement", ""),
            evidence=ev(dc, "requirement"),
        ))
    for iface in raw.get("interfaces", []):
        interfaces.append(Interface(
            counterparty=iface.get("counterparty", "Unknown"),
            description=iface.get("description", ""),
            direction=iface.get("direction", "coordinate"),
            evidence=ev(iface, "description"),
        ))
    for fl in raw.get("flags", []):
        sev_raw = fl.get("severity", "info")
        sev = Severity(sev_raw) if sev_raw in ("info", "coordinate", "rfi") else Severity.INFO
        flags.append(Flag(
            title=fl.get("title", "")[:100],
            detail=fl.get("detail", ""),
            severity=sev,
            evidence=ev(fl, "detail"),
            suggested_rfi=fl.get("suggested_rfi"),
        ))
    return items, criteria, interfaces, flags


# ---------------------------------------------------------------------------
# Project identity (unchanged heuristic — flagged low-confidence downstream)
# ---------------------------------------------------------------------------

PROJECT_LOCATION_RX = re.compile(r"Project\s+Location:\s*\n?\s*([A-Za-z][A-Za-z0-9 ,.'&-]{4,60})")


def extract_project_name_geometric(pdf_path: str) -> str | None:
    """Geometric fallback for multi-column CAD sheets where pdftotext's
    linear stream garbles the label/value pairing (same failure mode as the
    titleblock problem — proven fix is to read the actual page geometry)."""
    if fitz is None:
        return None
    try:
        pdf = fitz.open(pdf_path)
    except Exception:
        return None
    for i in range(min(5, len(pdf))):
        page = pdf[i]
        found = page.search_for("Project Location:")
        if not found:
            continue
        r = found[0]
        w = page.rect.width
        box = fitz.Rect(r.x0 - 20, r.y0, min(r.x0 + 350, w), r.y1 + 80)
        text = page.get_text("text", clip=box)
        m = PROJECT_LOCATION_RX.search(text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(" ,.")
    return None


def extract_project_name(docs: list["Document"]) -> str:
    for doc in docs:
        head = doc.page_text(1)
        for line in head.splitlines():
            line = line.strip()
            if 6 < len(line) < 70 and not line.isdigit():
                return line
    return "Untitled Project"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(package: dict[str, str], profile: ContractorProfile) -> ScopeReport:
    docs = [Document.load(p, label=Path(p).name) for p in package.values()]
    coverage = Coverage()

    # --- Pass 1: structure ---
    # Sheet discovery: try a compiled index page first (cheap, works well on
    # paginated multi-sheet sets like VCPS). Large-format sets often don't
    # have one, so always also run the titleblock scanner and merge results —
    # belt and suspenders rather than picking one strategy per document type.
    sheets: dict[str, SheetRef] = {}
    index_pages_total = 0
    for doc, path in zip(docs, package.values()):
        found, idx_pages = extract_sheet_index(doc)
        index_pages_total += idx_pages
        for s in found:
            sheets.setdefault(s.number, s)

        tb_found, tb_scanned, tb_parsed = extract_sheets_by_titleblock(doc, str(path))
        coverage.titleblock_pages_scanned += tb_scanned
        coverage.titleblock_pages_parsed += tb_parsed
        for s in tb_found:
            sheets.setdefault(s.number, s)
    coverage.sheet_index_pages_found = index_pages_total
    sheets = sorted(sheets.values(), key=lambda s: s.number)

    div21_sections = extract_division_sections(docs, "21")
    div21_blocks = extract_division_blocks(docs, "21")
    coverage.div21_blocks_found = len(div21_blocks)

    scope_blocks = locate_scope_blocks(docs)
    coverage.scope_blocks_found = len(scope_blocks)

    contract_category = scope_blocks[0][2] if scope_blocks else None

    # --- Pass 2: meaning ---
    client = _get_client()
    llm_available = client is not None

    all_items: list[ScopeItem] = []
    all_criteria: list[DesignCriterion] = []
    all_interfaces: list[Interface] = []
    all_flags: list[Flag] = []

    if llm_available:
        for doc, page, header, block in scope_blocks:
            raw = llm_extract(client, block, doc.label, page, coverage)
            i, c, iface, fl = _to_dataclasses(raw, doc.label, page, block, coverage)
            all_items += i; all_criteria += c; all_interfaces += iface; all_flags += fl

        for doc, page, block in div21_blocks:
            raw = llm_extract(client, block, doc.label, page, coverage)
            i, c, iface, fl = _to_dataclasses(raw, doc.label, page, block, coverage)
            all_items += i; all_criteria += c; all_interfaces += iface; all_flags += fl
    else:
        all_flags.append(Flag(
            title="LLM extraction unavailable — structural results only",
            detail=(f"ANTHROPIC_API_KEY not set (or anthropic SDK missing). Found "
                    f"{coverage.div21_blocks_found} Division 21 blocks and "
                    f"{coverage.scope_blocks_found} scope-of-work blocks but did not "
                    f"semantically extract scope items, design criteria, or interfaces "
                    f"from them. Sheets and section locations below are still structural "
                    f"(Pass 1) results."),
            severity=Severity.RFI,
            evidence=Evidence("system", 0, "no ANTHROPIC_API_KEY in environment"),
        ))

    # Division 21 gap check — only flag if structurally absent
    if not div21_sections:
        all_flags.append(Flag(
            title="Division 21 specifications not located",
            detail=("No Division 21 section headers were found in the uploaded "
                    "documents using the generalized header pattern. Either this "
                    "package doesn't include Division 21, or its section-header "
                    "format falls outside the tolerant pattern — worth a spot check."),
            severity=Severity.RFI,
            evidence=Evidence("package", 0, "Division 21 absent from upload"),
            suggested_rfi=("Division 21 — Fire Suppression specifications do not appear "
                           "in the bid documents issued to us. Please confirm the correct "
                           "volume of the project manual containing Division 21."),
        ))

    project_name = None
    for p in package.values():
        project_name = extract_project_name_geometric(str(p))
        if project_name:
            break
    if not project_name:
        project_name = extract_project_name(docs)

    return ScopeReport(
        project=project_name,
        contractor=profile.company,
        contract_category=contract_category,
        documents=[{"file": d.label, "pages": len(d.pages)} for d in docs],
        sheets=sheets,
        scope_items=all_items,
        spec_refs=[],  # superseded by div21_sections + LLM-extracted design_criteria
        interfaces=all_interfaces,
        flags=all_flags,
        div21_sections=div21_sections,
        design_criteria=all_criteria,
        coverage=coverage,
        llm_available=llm_available,
    )


if __name__ == "__main__":
    import sys
    pkg = {f"doc{i}": p for i, p in enumerate(sys.argv[1:])}
    report = analyze(pkg, ContractorProfile(company="Demo Fire Protection"))
    print(report.to_json())
