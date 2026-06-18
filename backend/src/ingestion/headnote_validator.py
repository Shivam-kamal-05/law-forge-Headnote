"""Post-extraction validation for LLM-produced headnotes.

Checks for the most dangerous failure mode — a subject-matter mismatch
where the headnote describes a precedent cited inside the judgment instead
of the case actually decided. Flagged records are still returned to the
client; `needs_review` signals that manual verification is recommended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingestion.structured_extractor import StructuredMetadata
from src.schemas.headnote import LLMExtraction

_NAME_STOPWORDS: frozenset[str] = frozenset({
    "versus", "anr", "ors", "another", "others", "etc",
    "state", "union", "india", "govt", "government",
    "ltd", "limited", "pvt", "private", "company",
    "the", "and", "for", "through", "represented",
})

_WORD = re.compile(r"[a-z]+")
_PARA_MARKER = re.compile(r"^\s*(\d{1,4})[.)]\s", re.MULTILINE)

_NAME_MATCH_THRESHOLD = 0.34


@dataclass(slots=True)
class ValidationReport:
    needs_review: bool = False
    reasons: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        self.reasons.append(reason)
        self.needs_review = True


def significant_tokens(name: str | None) -> set[str]:
    if not name:
        return set()
    return {
        tok
        for tok in _WORD.findall(name.lower())
        if len(tok) >= 3 and tok not in _NAME_STOPWORDS
    }


def names_align(name_a: str | None, name_b: str | None) -> bool:
    ta = significant_tokens(name_a)
    tb = significant_tokens(name_b)
    if not ta or not tb:
        return True
    overlap = len(ta & tb) / min(len(ta), len(tb))
    return overlap >= _NAME_MATCH_THRESHOLD


def max_paragraph_number(judgment_text: str) -> int:
    nums = [int(m.group(1)) for m in _PARA_MARKER.finditer(judgment_text)]
    return max(nums) if nums else 0


def validate_extraction(
    *,
    structured: StructuredMetadata,
    llm: LLMExtraction,
    judgment_text: str,
) -> ValidationReport:
    """Validate one LLM extraction against its source metadata and text."""
    report = ValidationReport()

    meta_name = structured.case_name
    text_name = (llm.case_name_in_text or "").strip()

    if meta_name and not text_name:
        report.add(
            "The model did not echo the case name from the judgment text "
            "(case_name_in_text empty) — alignment could not be verified."
        )
    elif meta_name and text_name and not names_align(meta_name, text_name):
        report.add(
            f"Case-name mismatch: document metadata identifies the case as "
            f"'{meta_name}', but the model read the judgment as concerning "
            f"'{text_name}'. The headnote may describe the wrong case."
        )

    max_para = max_paragraph_number(judgment_text)
    if max_para > 0:
        out_of_range = sorted(
            {p for section in llm.head_note for p in section.paragraphs if p > max_para}
        )
        if out_of_range:
            report.add(
                f"Paragraph references {out_of_range} fall outside the "
                f"judgment (last numbered paragraph is {max_para})."
            )

    if not llm.head_note:
        report.add("No headnote heads were extracted.")
    else:
        missing_holding = sum(1 for s in llm.head_note if not s.holding.strip())
        if missing_holding:
            report.add(
                f"{missing_holding} of {len(llm.head_note)} headnote head(s) "
                f"have no operative holding."
            )

    return report
