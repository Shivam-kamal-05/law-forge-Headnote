"""Pydantic schemas for headnote extraction.

Three layers:
* `HeadnoteSection` — one structured head of an Indian-style headnote.
* `LLMExtraction`   — the subset the LLM is asked to produce via tool call.
* `HeadnoteResponse` / `ExtractionError` — the API wire format.
"""

from __future__ import annotations

import html
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.schemas.enums import (
    BENCH_TYPE_LABELS,
    OUTCOME_LABELS,
    BenchType,
    DecisionType,
    Outcome,
)

ShortText = Annotated[str, StringConstraints(min_length=0, max_length=10_000, strip_whitespace=True)]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DDMMYYYY = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")


def _to_iso(date_str: str | None) -> str | None:
    if not date_str:
        return None
    s = date_str.strip()
    if _ISO_DATE.match(s):
        return s
    if (m := _DDMMYYYY.match(s)) is not None:
        d, mo, y = m.groups()
        return f"{y}-{mo}-{d}"
    return None


def format_para_refs(paras: list[int]) -> str:
    """Collapse a paragraph list into a compact range string e.g. '12, 14-16'."""
    nums = sorted({p for p in paras if p >= 1})
    if not nums:
        return ""
    runs: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    runs.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(runs)


# ---------------------------------------------------------------------------
# HeadnoteSection
# ---------------------------------------------------------------------------


class HeadnoteSection(BaseModel):
    """One 'head' of an Indian-style (SCC / Law Finder) headnote.

    Catchword cascade → operative holding → optional discussion →
    supporting paragraph numbers → cited authorities.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    catchwords: Annotated[str, StringConstraints(min_length=3, max_length=600)] = Field(
        ...,
        description=(
            "Catchword cascade: governing statute + provision first, then a "
            "' — '-separated drill-down to the precise issue."
        ),
    )
    holding: Annotated[str, StringConstraints(min_length=0, max_length=1200)] = Field(
        default="",
        description="Single-sentence operative holding (ratio decidendi).",
    )
    discussion: Annotated[str, StringConstraints(min_length=0, max_length=3000)] = Field(
        default="",
        description="Optional 1-3 sentence elaboration of the reasoning.",
    )
    paragraphs: list[int] = Field(
        default_factory=list,
        description="Numbered paragraphs of THIS judgment that support the head.",
    )
    cited_authorities: list[str] = Field(
        default_factory=list,
        description="Precedents the court relied on for this point.",
    )

    @field_validator("paragraphs")
    @classmethod
    def _clean_paragraphs(cls, v: list[int]) -> list[int]:
        return sorted({p for p in v if p >= 1})

    @field_validator("cited_authorities")
    @classmethod
    def _dedupe_authorities(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in v:
            cleaned = raw.strip()
            if cleaned:
                seen.setdefault(cleaned, None)
        return list(seen.keys())

    def to_html(self) -> str:
        parts: list[str] = [f"<p><b>{html.escape(self.catchwords)}</b></p>"]
        if self.holding:
            parts.append(f"<p><b>Held:</b> {html.escape(self.holding)}</p>")
        if self.discussion:
            parts.append(f"<p>{html.escape(self.discussion)}</p>")
        if self.paragraphs:
            parts.append(f"<p><b>Paras:</b> {html.escape(format_para_refs(self.paragraphs))}</p>")
        if self.cited_authorities:
            joined = "; ".join(html.escape(a) for a in self.cited_authorities)
            parts.append(f"<p><b>Relied on:</b> {joined}</p>")
        return "".join(parts)

    def to_plain_text(self) -> str:
        bits = [self.catchwords, self.holding, self.discussion]
        if self.cited_authorities:
            bits.append("Relied on: " + "; ".join(self.cited_authorities))
        return " ".join(b for b in bits if b)


# ---------------------------------------------------------------------------
# LLMExtraction — what the model returns via tool call
# ---------------------------------------------------------------------------


class LLMExtraction(BaseModel):
    """Substantive fields produced by the LLM. Structural metadata (court,
    judge, date, citations) comes from the deterministic regex extractor
    and is merged in by the API layer."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    case_name_in_text: str = Field(
        default="",
        max_length=500,
        description=(
            "The case name exactly as it appears in the court's own text — "
            "used to verify the headnote describes the correct judgment."
        ),
    )
    subject_classification: Annotated[str, StringConstraints(min_length=2, max_length=120)] = Field(
        ..., description="Branch of law (Criminal / Constitutional / Civil / ...)."
    )
    head_note: list[HeadnoteSection] = Field(
        default_factory=list,
        description="Structured headnote heads, one per distinct point of law.",
    )
    brief_facts: ShortText = Field(default="", description="2-3 sentence factual background.")
    held: ShortText = Field(default="", description="1-2 sentence operative outcome.")
    petitioner_argument: ShortText = Field(default="")
    respondent_argument: ShortText = Field(default="")
    statutes: list[str] = Field(default_factory=list)
    statutes_with_sections: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    importance_score: int = Field(default=3, ge=1, le=5)
    outcome: Outcome = Outcome.UNKNOWN
    decision_type: DecisionType = DecisionType.JUDGMENT

    # Fallback identity fields — filled by LLM only when the regex layer missed them.
    title: str | None = Field(default=None, max_length=500)
    citation: str | None = Field(default=None, max_length=400)
    court: str | None = Field(default=None, max_length=200)
    judge_name: str | None = Field(default=None, max_length=500)
    decision_date: str | None = Field(default=None, description="DD.MM.YYYY or YYYY-MM-DD")
    case_number: str | None = Field(default=None, max_length=300)

    @field_validator("statutes", "statutes_with_sections", "keywords")
    @classmethod
    def _strip_dedupe(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in v:
            cleaned = raw.strip()
            if cleaned:
                seen.setdefault(cleaned, None)
        return list(seen.keys())

    @field_validator("decision_date")
    @classmethod
    def _parse_date(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        if _to_iso(v) is None:
            raise ValueError(f"decision_date must be DD.MM.YYYY or YYYY-MM-DD, got {v!r}")
        return v.strip()


# ---------------------------------------------------------------------------
# API wire format
# ---------------------------------------------------------------------------


class HeadnoteResponse(BaseModel):
    """Full extraction result returned to the client for one PDF."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Identity
    document_id: str = Field(..., description="MD5 of the source PDF.")
    filename: str

    # Merged metadata (structured extractor + LLM fallbacks)
    case_name: str | None
    court: str | None
    bench: str | None
    bench_type: int
    bench_type_label: str
    decision_date: str | None = Field(description="ISO YYYY-MM-DD.")
    case_number: str | None
    parallel_citations: list[str]
    petitioner: str | None
    respondent: str | None

    # LLM substantive output
    subject_classification: str
    head_note: list[HeadnoteSection]
    brief_facts: str
    held: str
    petitioner_argument: str
    respondent_argument: str
    statutes: list[str]
    statutes_with_sections: list[str]
    keywords: list[str]
    importance_score: int
    outcome: int
    outcome_label: str
    decision_type: int
    decision_type_label: str
    cases_referred: list[str]

    # Validation
    needs_review: bool
    review_reasons: list[str]

    # Provenance
    used_ocr: bool
    judgment_marker_found: bool
    processing_time_ms: int

    @classmethod
    def build(
        cls,
        *,
        document_id: str,
        filename: str,
        structured: object,  # StructuredMetadata
        llm: LLMExtraction,
        used_ocr: bool,
        judgment_marker_found: bool,
        needs_review: bool,
        review_reasons: list[str],
        processing_time_ms: int,
    ) -> HeadnoteResponse:
        from src.schemas.enums import infer_bench_type
        from src.ingestion.structured_extractor import StructuredMetadata

        s: StructuredMetadata = structured  # type: ignore[assignment]

        case_name = s.case_name or llm.title
        court = s.court or llm.court
        bench = s.bench or llm.judge_name
        bench_type = int(infer_bench_type(bench))
        decision_date_iso = s.decision_date_iso or (
            _to_iso(llm.decision_date) if llm.decision_date else None
        )
        case_number = s.case_number or llm.case_number
        citations = list(s.parallel_citations)
        if not citations and llm.citation:
            citations = [llm.citation]

        cases_referred_raw = [
            f"{c.case_name}, {c.citation}" for c in s.cases_referred
        ]

        return cls(
            document_id=document_id,
            filename=filename,
            case_name=case_name,
            court=court,
            bench=bench,
            bench_type=bench_type,
            bench_type_label=BENCH_TYPE_LABELS.get(bench_type, "Unknown"),
            decision_date=decision_date_iso,
            case_number=case_number,
            parallel_citations=citations,
            petitioner=s.petitioner,
            respondent=s.respondent,
            subject_classification=llm.subject_classification,
            head_note=list(llm.head_note),
            brief_facts=llm.brief_facts,
            held=llm.held,
            petitioner_argument=llm.petitioner_argument,
            respondent_argument=llm.respondent_argument,
            statutes=list(llm.statutes),
            statutes_with_sections=list(llm.statutes_with_sections),
            keywords=list(llm.keywords),
            importance_score=llm.importance_score,
            outcome=int(llm.outcome),
            outcome_label=OUTCOME_LABELS.get(int(llm.outcome), "Unknown"),
            decision_type=int(llm.decision_type),
            decision_type_label="Judgment" if int(llm.decision_type) == 2 else "Order",
            cases_referred=cases_referred_raw,
            needs_review=needs_review,
            review_reasons=review_reasons,
            used_ocr=used_ocr,
            judgment_marker_found=judgment_marker_found,
            processing_time_ms=processing_time_ms,
        )


class ExtractionError(BaseModel):
    """Returned when a single file fails — allows partial success in batch requests."""

    filename: str
    document_id: str | None = None
    error_code: str
    message: str
    processing_time_ms: int
