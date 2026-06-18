"""Core headnote extraction endpoint.

POST /api/v1/extract
--------------------
Accepts one or more PDF uploads via multipart/form-data. Each file goes
through the full pipeline:

    parse (PyMuPDF → pdfplumber → OCR)
    → strip publisher chrome + split at JUDGMENT
    → deterministic structured-extract (regex)
    → LLM extraction (Anthropic tool call, forced)
    → validation (case-name alignment + para-range checks)
    → HeadnoteResponse

Up to `settings.ingestion.concurrency` files are processed concurrently.
A file that fails (bad PDF, LLM error, etc.) returns an `ExtractionError`
in its slot — the rest of the batch is unaffected.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Union

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.dependencies import LLMClientDep, RequireAuth, SettingsDep
from src.core.exceptions import LawLensError
from src.core.logging import get_logger
from src.ingestion.headnote_validator import validate_extraction
from src.ingestion.parser import PDFParser
from src.ingestion.structured_extractor import extract_metadata
from src.schemas.headnote import ExtractionError, HeadnoteResponse

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extract"])


def _metadata_for_prompt(structured: Any) -> dict[str, Any]:
    return {
        "title": structured.case_name,
        "citation": structured.parallel_citations[0] if structured.parallel_citations else None,
        "court": structured.court,
        "judge_name": structured.bench,
        "decision_date": structured.date,
        "case_number": structured.case_number,
        "petitioner": structured.petitioner,
        "respondent": structured.respondent,
    }


async def _process_one(
    upload: UploadFile,
    *,
    llm: Any,
    settings: Any,
) -> Union[HeadnoteResponse, ExtractionError]:
    """Full pipeline for a single uploaded PDF."""
    filename = upload.filename or "document.pdf"
    started = time.perf_counter()

    content = await upload.read()
    elapsed_ms = lambda: int((time.perf_counter() - started) * 1000)

    if not content:
        return ExtractionError(
            filename=filename,
            error_code="empty_file",
            message="Uploaded file is empty.",
            processing_time_ms=elapsed_ms(),
        )

    if len(content) > settings.ingestion.max_file_bytes:
        limit_mb = settings.ingestion.max_file_bytes // (1024 * 1024)
        return ExtractionError(
            filename=filename,
            error_code="payload_too_large",
            message=f"File exceeds {limit_mb} MiB limit.",
            processing_time_ms=elapsed_ms(),
        )

    doc_id = hashlib.md5(content).hexdigest()

    # Write to a temp file so PyMuPDF / pdfplumber can seek it.
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        parser = PDFParser(settings.ingestion)

        try:
            parsed = await parser.parse_and_clean(tmp_path)
        except LawLensError as exc:
            log.warning("extract.parse_failed", filename=filename, error=exc.message)
            return ExtractionError(
                filename=filename,
                document_id=doc_id,
                error_code=exc.error_code,
                message=exc.message,
                processing_time_ms=elapsed_ms(),
            )

        cleaned = parsed.cleaned
        if cleaned is None or not cleaned.judgment_text.strip():
            return ExtractionError(
                filename=filename,
                document_id=doc_id,
                error_code="empty_judgment_text",
                message="No court text found after stripping publisher chrome. "
                        "Ensure the PDF contains a valid Indian court judgment.",
                processing_time_ms=elapsed_ms(),
            )

        structured = extract_metadata(cleaned.metadata_block)

        try:
            llm_result = await llm.extract_one(
                custom_id=doc_id,
                filename=filename,
                text=cleaned.judgment_text,
                metadata=_metadata_for_prompt(structured),
            )
        except LawLensError as exc:
            log.error("extract.llm_failed", filename=filename, doc_id=doc_id, error=exc.message)
            return ExtractionError(
                filename=filename,
                document_id=doc_id,
                error_code=exc.error_code,
                message=f"LLM extraction failed: {exc.message}",
                processing_time_ms=elapsed_ms(),
            )

        report = validate_extraction(
            structured=structured,
            llm=llm_result,
            judgment_text=cleaned.judgment_text,
        )

        if report.needs_review:
            log.warning(
                "extract.flagged_for_review",
                filename=filename,
                doc_id=doc_id,
                reasons=report.reasons,
            )

        response = HeadnoteResponse.build(
            document_id=doc_id,
            filename=filename,
            structured=structured,
            llm=llm_result,
            used_ocr=parsed.used_ocr,
            judgment_marker_found=cleaned.judgment_marker_found,
            needs_review=report.needs_review,
            review_reasons=report.reasons,
            processing_time_ms=elapsed_ms(),
        )

        log.info(
            "extract.success",
            filename=filename,
            doc_id=doc_id,
            heads=len(llm_result.head_note),
            subject=llm_result.subject_classification,
            importance=llm_result.importance_score,
            used_ocr=parsed.used_ocr,
            took_ms=response.processing_time_ms,
        )

        return response

    except Exception as exc:  # noqa: BLE001
        log.exception("extract.unexpected_error", filename=filename, error=str(exc))
        return ExtractionError(
            filename=filename,
            document_id=doc_id if "doc_id" in dir() else None,
            error_code="unexpected_error",
            message=f"Unexpected error: {type(exc).__name__}: {exc}",
            processing_time_ms=elapsed_ms(),
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@router.post(
    "/extract",
    response_model=list[Union[HeadnoteResponse, ExtractionError]],
    summary="Extract structured headnotes from one or more PDF judgments",
    dependencies=[RequireAuth],
)
async def extract_headnotes(
    files: list[UploadFile] = File(..., description="PDF judgment file(s). Up to 10 per request."),
    llm: LLMClientDep = None,  # type: ignore[assignment]
    settings: SettingsDep = None,  # type: ignore[assignment]
) -> list[Union[HeadnoteResponse, ExtractionError]]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    max_files = settings.ingestion.max_files_per_request
    if len(files) > max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {max_files} files per request. Received {len(files)}.",
        )

    sem = asyncio.Semaphore(settings.ingestion.concurrency)

    async def process_with_sem(upload: UploadFile) -> Union[HeadnoteResponse, ExtractionError]:
        async with sem:
            return await _process_one(upload, llm=llm, settings=settings)

    results = await asyncio.gather(*[process_with_sem(f) for f in files])
    return list(results)
