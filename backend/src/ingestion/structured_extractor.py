"""Deterministic regex-based extraction of structured metadata from the
pre-JUDGMENT block of an Indian court PDF.

These fields are cheaper, faster, and more accurate to extract here than
to ask the LLM — the layout is predictable. We only use the LLM for
the substantive headnote (brief facts, held, legal propositions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SINGLE_CITATION = re.compile(
    r"""
    \(?\d{4}\)?
    \s*
    (?:\(\d+\)|\s\d+)?
    \s+
    [A-Z][A-Za-z\.\s\(\)&/]{1,40}?
    \s+
    \d+[A-Z]?
    """,
    re.VERBOSE,
)

_DOC_ID_LINE = re.compile(r"Law\s*Finder\s*Doc\s*Id\s*#\s*(\d+)", re.IGNORECASE)

_COURT_LINE = re.compile(
    r"^\s*(SUPREME COURT OF INDIA|HIGH COURT OF [A-Z][A-Za-z\.,&\s\-]+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_BENCH_LINE = re.compile(
    r"Before\s*[:\-]+\s*(.+?)(?=\n\s*\n|Civil\s+Appeal|Criminal\s+Appeal|"
    r"S\.?L\.?P|Special\s+Leave|Writ\s+Petition|Transfer|Review|Letters\s+Patent|"
    r"Contempt|Suo\s+Motu|Election\s+Petition|D/d\.)",
    re.DOTALL | re.IGNORECASE,
)

_CASE_NUMBER = re.compile(
    r"((?:Civil|Criminal|Special\s+Leave|Writ|Transfer|Review|Letters\s+Patent|"
    r"Contempt|Election|Suo\s+Motu|Curative|Original|Tax|Arbitration)\s+"
    r"(?:Appeal|Petition|Application|Revision|Case|Suit|Reference)s?"
    r"(?:\s*\([A-Za-z\.]+\))?"
    r"\s+No?s?\.?\s*[\d,\s\-]+(?:\s+of\s+\d{4})"
    r"(?:\s*\(Arising\s+out\s+of\s+[^)]+\))?"
    r")",
    re.IGNORECASE,
)

_DATE_DD = re.compile(r"D\s*/\s*d\.\s*(\d{1,2})\.(\d{1,2})\.(\d{4})", re.IGNORECASE)
_DATE_GENERIC = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})\b")

_CAUSE_TITLE_VERSUS = re.compile(
    r"(?P<petitioner>[A-Z][^\n]{2,200}?)\s*[-–]?\s*(?:Appellants?|Petitioners?|Plaintiffs?)\s*\n"
    r"\s*(?:Versus|VS\.?|v\.)\s*\n"
    r"\s*(?P<respondent>[^\n]{2,200}?)\s*[-–]?\s*(?:Respondents?|Defendants?)",
    re.IGNORECASE | re.DOTALL,
)

_CAUSE_TITLE_INLINE = re.compile(
    r"^([A-Z][A-Za-z0-9\.\s,&\(\)\-']{2,200}?)\s+v\.\s+([A-Za-z0-9\.\s,&\(\)\-']{2,200})",
    re.MULTILINE,
)

_COUNSEL_APPELLANT = re.compile(
    r"For\s+the\s+Appellants?\s*[:\-]+\s*(.+?)(?=For\s+the\s+Respondents?|JUDGMENT|ORDER|Cases?\s+Referred|\n\s*\n[A-Z])",
    re.DOTALL | re.IGNORECASE,
)
_COUNSEL_RESPONDENT = re.compile(
    r"For\s+the\s+Respondents?\s*[:\-]+\s*(.+?)(?=For\s+the\s+|JUDGMENT|ORDER|Cases?\s+Referred|\n\s*\n[A-Z])",
    re.DOTALL | re.IGNORECASE,
)

_CASES_REFERRED_BLOCK = re.compile(
    r"Cases?\s+Referred\s*[:\-]+\s*(.+?)(?=JUDGMENT|ORDER|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_CITED_CASE_LINE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9\.\s,&\(\)\-']{4,200}?\s+v\.?\s+[A-Za-z0-9\.\s,&\(\)\-']{2,200}?),\s*(.+?\.)\s*$",
    re.MULTILINE,
)


@dataclass(slots=True)
class StructuredMetadata:
    """Everything pulled deterministically from the metadata block."""

    law_finder_doc_id: str | None = None
    court: str | None = None
    bench: str | None = None
    bench_type: int = 0
    case_number: str | None = None
    date: str | None = None
    decision_date_iso: str | None = None
    case_name: str | None = None
    petitioner: str | None = None
    respondent: str | None = None
    parallel_citations: list[str] = field(default_factory=list)
    counsel_for_appellants: list[str] = field(default_factory=list)
    counsel_for_respondents: list[str] = field(default_factory=list)
    counsel_for_appellants_string: str = ""
    counsel_for_respondents_string: str = ""
    cases_referred: list[CitedCase] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class CitedCase:
    case_name: str
    citation: str


def extract_metadata(metadata_block: str) -> StructuredMetadata:
    """Parse the pre-JUDGMENT region. Each field is best-effort — missing
    fields stay None / [] rather than raising."""
    if not metadata_block.strip():
        return StructuredMetadata()

    out = StructuredMetadata()

    if (m := _DOC_ID_LINE.search(metadata_block)) is not None:
        out.law_finder_doc_id = m.group(1)

    if (m := _COURT_LINE.search(metadata_block)) is not None:
        out.court = _normalize_whitespace(m.group(1))

    if (m := _BENCH_LINE.search(metadata_block)) is not None:
        out.bench = _normalize_whitespace(m.group(1).rstrip(",. "))

    if (m := _CASE_NUMBER.search(metadata_block)) is not None:
        out.case_number = _normalize_whitespace(m.group(1))

    out.date = _extract_date(metadata_block)
    out.decision_date_iso = _to_iso(out.date)

    pet, resp = _extract_cause_title(metadata_block)
    out.petitioner = pet
    out.respondent = resp
    if pet and resp:
        out.case_name = f"{pet} v. {resp}"

    raw_appellant = _first_match(_COUNSEL_APPELLANT, metadata_block)
    raw_respondent = _first_match(_COUNSEL_RESPONDENT, metadata_block)
    out.counsel_for_appellants = _split_counsel(raw_appellant)
    out.counsel_for_respondents = _split_counsel(raw_respondent)
    if raw_appellant:
        out.counsel_for_appellants_string = "- " + _normalize_whitespace(raw_appellant)
    if raw_respondent:
        out.counsel_for_respondents_string = "- " + _normalize_whitespace(raw_respondent)

    if out.bench:
        from src.schemas.enums import infer_bench_type
        out.bench_type = int(infer_bench_type(out.bench))

    out.parallel_citations = _extract_parallel_citations(metadata_block)
    out.cases_referred = _extract_cases_referred(metadata_block)

    return out


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _to_iso(dd_mm_yyyy: str | None) -> str | None:
    if not dd_mm_yyyy:
        return None
    parts = dd_mm_yyyy.split(".")
    if len(parts) != 3:
        return None
    d, m, y = parts
    return f"{y}-{m}-{d}"


def _extract_date(text: str) -> str | None:
    m = _DATE_DD.search(text)
    if m is not None:
        d, mo, y = m.groups()
        return f"{int(d):02d}.{int(mo):02d}.{int(y):04d}"

    candidates: list[tuple[int, int, int]] = []
    for d, mo, y in _DATE_GENERIC.findall(text):
        di, mi, yi = int(d), int(mo), int(y)
        if 1 <= di <= 31 and 1 <= mi <= 12 and 1900 <= yi <= 2100:
            candidates.append((yi, mi, di))
    if not candidates:
        return None
    yi, mi, di = max(candidates)
    return f"{di:02d}.{mi:02d}.{yi:04d}"


def _extract_cause_title(text: str) -> tuple[str | None, str | None]:
    if (m := _CAUSE_TITLE_VERSUS.search(text)) is not None:
        return (
            _normalize_whitespace(m.group("petitioner")),
            _normalize_whitespace(m.group("respondent")),
        )
    if (m := _CAUSE_TITLE_INLINE.search(text)) is not None:
        return (
            _normalize_whitespace(m.group(1)),
            _normalize_whitespace(m.group(2)),
        )
    return (None, None)


def _split_counsel(raw: str | None) -> list[str]:
    if not raw:
        return []
    cleaned = _normalize_whitespace(raw).rstrip(".")
    cleaned = re.sub(r",?\s*Advocates?\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*Senior\s+Advocates?\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    parts = [p.strip(" .,;") for p in re.split(r"[,;]", cleaned)]
    return [
        p for p in parts
        if p and not re.fullmatch(r"(Sr\.?\s*Adv(?:ocate)?\.?|Advocates?\.?|Sr\.?)", p, re.IGNORECASE)
    ]


def _extract_parallel_citations(text: str) -> list[str]:
    doc_id_match = _DOC_ID_LINE.search(text)
    start = doc_id_match.end() if doc_id_match else 0
    tail = text[start:]
    block_end_re = re.compile(
        r"^\s*(?:SUPREME\s+COURT\s+OF\s+INDIA|HIGH\s+COURT\s+OF|Before\s*[:\-])",
        re.IGNORECASE | re.MULTILINE,
    )
    bm = block_end_re.search(tail)
    block = tail[: bm.start()] if bm else tail[:2000]

    raw_tokens = re.split(r"\s*:\s*|\n", block)
    citations: list[str] = []
    for tok in raw_tokens:
        tok = _normalize_whitespace(tok)
        if not tok or tok.isdigit():
            continue
        if _SINGLE_CITATION.search(tok):
            citations.append(tok)

    seen: dict[str, None] = {}
    for c in citations:
        seen.setdefault(c, None)
    return list(seen.keys())


def _extract_cases_referred(text: str) -> list[CitedCase]:
    block_match = _CASES_REFERRED_BLOCK.search(text)
    if block_match is None:
        return []
    block = block_match.group(1)
    out: list[CitedCase] = []
    for case_match in _CITED_CASE_LINE.finditer(block):
        name = _normalize_whitespace(case_match.group(1))
        citation = _normalize_whitespace(case_match.group(2)).rstrip(".")
        if name and citation:
            out.append(CitedCase(case_name=name, citation=citation))
    return out
