"""Publisher-chrome redaction — strips Law Finder brand / doc-id from text.

Applied at API egress to ensure proprietary publisher identifiers never
reach API consumers. The in-memory text is never mutated.
"""

from __future__ import annotations

import re
from typing import overload

_DOC_ID_RE = re.compile(
    r"""
    \b
    law\s*finder
    (?:\s*[-–—]?\s*)
    (?:doc(?:ument)?\s*id\s*)?
    [#:]?\s*\d+
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LICENSED_TO_RE = re.compile(
    r"Licensed\s*to\s*:[^\n|]*?(?=$|\n|\|)",
    re.IGNORECASE,
)

_BRAND_RE = re.compile(
    r"""
    (?:
        \blawfinder(?:\.[a-z]{2,})+\b
      | \blaw\s*finder\b
        (?:\s+publishers?)?
        (?:\s+pvt\.?\s*ltd\.?)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CLEAN_WS_RE = re.compile(r"[ \t]{2,}")
_EMPTY_BRACKET_RE = re.compile(r"\(\s*\)|\[\s*\]")
_DOC_ID_VALUE_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _doc_id_value_pattern(doc_id: str) -> re.Pattern[str]:
    cached = _DOC_ID_VALUE_RE_CACHE.get(doc_id)
    if cached is not None:
        return cached
    pattern = re.compile(rf"(?<!\d){re.escape(doc_id)}(?!\d)")
    _DOC_ID_VALUE_RE_CACHE[doc_id] = pattern
    return pattern


def redact_text(value: str, *, doc_id: str | None = None) -> str:
    if not value:
        return value
    out = _LICENSED_TO_RE.sub("", value)
    out = _DOC_ID_RE.sub("", out)
    out = _BRAND_RE.sub("", out)
    if doc_id:
        out = _doc_id_value_pattern(doc_id).sub("", out)
    out = _EMPTY_BRACKET_RE.sub("", out)
    out = _CLEAN_WS_RE.sub(" ", out)
    return out.strip()


@overload
def redact_strings(value: str, *, doc_id: str | None = ...) -> str: ...
@overload
def redact_strings(value: list[str], *, doc_id: str | None = ...) -> list[str]: ...


def redact_strings(value: str | list[str], *, doc_id: str | None = None) -> str | list[str]:
    if isinstance(value, list):
        return [redact_text(v, doc_id=doc_id) for v in value]
    return redact_text(value, doc_id=doc_id)


def contains_publisher_chrome(value: str) -> bool:
    return bool(
        _BRAND_RE.search(value)
        or _DOC_ID_RE.search(value)
        or _LICENSED_TO_RE.search(value)
    )
