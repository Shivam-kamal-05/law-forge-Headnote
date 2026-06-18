"""Strip publisher chrome and split text at the JUDGMENT/ORDER boundary.

Indian judgment PDFs from publishers like Law Finder, SCC, AIR share a
predictable layout — the copyrighted publisher headnote block sits BEFORE
the JUDGMENT marker. We split there so the LLM only ever sees the court's
own text, eliminating plagiarism risk at the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^.{0,40}(Law Finder|Manupatra|SCC Online|Indian Kanoon|AIR\s*Online).*?Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
)

_FOOTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(Law Finder|Manupatra|SCC Online|Indian Kanoon)\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s*$", re.MULTILINE),
)

_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^.*Licensed\s*to\s*:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Product\s*S\.?No\.?\s*\d+.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^This judg(?:e)?ment ranked\s+\d+\s+in the hitlist\.?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:©|Copyright\s+©?).*$", re.IGNORECASE | re.MULTILINE),
)

_JUDGMENT_MARKER = re.compile(
    r"^\s*("
    r"J\s*U\s*D\s*G\s*M\s*E\s*N\s*T"
    r"|O\s*R\s*D\s*E\s*R"
    r"|JUDGMENT"
    r"|ORDER"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(slots=True, frozen=True)
class CleanedDocument:
    raw_text: str
    """Original text with only page chrome stripped."""

    metadata_block: str
    """Pre-JUDGMENT region — fed to the deterministic regex extractor."""

    judgment_text: str
    """Post-JUDGMENT court text — the ONLY thing sent to the LLM."""

    judgment_marker_found: bool
    """False when no JUDGMENT/ORDER marker was found; full doc used as fallback."""


def strip_page_chrome(text: str) -> str:
    cleaned = text
    for pat in (*_HEADER_PATTERNS, *_FOOTER_PATTERNS, *_NOISE_PATTERNS):
        cleaned = pat.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_at_judgment(text: str) -> tuple[str, str, bool]:
    """Locate JUDGMENT/ORDER boundary. Returns (metadata_block, judgment_text, found)."""
    match = _JUDGMENT_MARKER.search(text)
    if match is None:
        return ("", text, False)
    head = text[: match.start()].rstrip()
    tail = text[match.end() :].lstrip()
    return (head, tail, True)


def clean_and_split(raw_text: str) -> CleanedDocument:
    chromeless = strip_page_chrome(raw_text)
    metadata_block, judgment_text, marker_found = split_at_judgment(chromeless)
    return CleanedDocument(
        raw_text=chromeless,
        metadata_block=metadata_block,
        judgment_text=judgment_text,
        judgment_marker_found=marker_found,
    )
