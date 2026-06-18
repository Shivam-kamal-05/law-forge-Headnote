"""PDF text extraction with OCR fallback for scanned judgments.

Strategy:
1. PyMuPDF  — fastest, best layout retention.
2. pdfplumber — fallback when PyMuPDF yields too little text.
3. pytesseract — last resort OCR for scanned / image-only PDFs.

Older Indian Supreme Court / High Court PDFs (2011-2015) are frequently
scanned images, so OCR is not optional for full corpus coverage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from src.core.config import IngestionSettings, get_settings
from src.core.exceptions import OCRFailureError, PDFParseError
from src.core.logging import get_logger
from src.ingestion.text_cleaner import CleanedDocument, clean_and_split

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ParsedPDF:
    text: str
    page_count: int
    used_ocr: bool
    cleaned: CleanedDocument | None = None


class PDFParser:
    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._settings = settings or get_settings().ingestion

    async def parse_and_clean(self, path: Path) -> ParsedPDF:
        """Parse PDF then run publisher-chrome / JUDGMENT-marker cleaner."""
        parsed = await self.parse(path)
        cleaned = clean_and_split(parsed.text)
        return ParsedPDF(
            text=parsed.text,
            page_count=parsed.page_count,
            used_ocr=parsed.used_ocr,
            cleaned=cleaned,
        )

    async def parse(self, path: Path) -> ParsedPDF:
        if not path.exists():
            raise PDFParseError(f"PDF does not exist: {path}")

        primary = await asyncio.to_thread(self._parse_pymupdf, path)
        if primary and len(primary.text) >= self._settings.min_text_length:
            return primary

        log.warning(
            "parser.primary_low_yield",
            path=str(path),
            chars=len(primary.text) if primary else 0,
        )

        secondary = await asyncio.to_thread(self._parse_pdfplumber, path)
        if secondary and len(secondary.text) >= self._settings.min_text_length:
            return secondary

        if not self._settings.ocr_enabled:
            best = secondary or primary
            if not best or not best.text.strip():
                raise PDFParseError(f"No extractable text in {path} (OCR disabled).")
            return best

        log.info("parser.fallback_to_ocr", path=str(path))
        return await asyncio.to_thread(self._parse_ocr, path)

    @staticmethod
    def _parse_pymupdf(path: Path) -> ParsedPDF | None:
        import fitz

        try:
            doc = fitz.open(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("parser.pymupdf_failed", path=str(path), error=str(exc))
            return None

        try:
            pages: list[str] = []
            for page in doc:
                pages.append(page.get_text("text"))  # type: ignore[no-untyped-call]
            text = "\n".join(pages).strip()
            return ParsedPDF(text=text, page_count=doc.page_count, used_ocr=False)
        finally:
            doc.close()

    @staticmethod
    def _parse_pdfplumber(path: Path) -> ParsedPDF | None:
        import pdfplumber

        try:
            with pdfplumber.open(path) as pdf:
                pages = [(p.extract_text() or "") for p in pdf.pages]
                text = "\n".join(pages).strip()
                return ParsedPDF(text=text, page_count=len(pdf.pages), used_ocr=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("parser.pdfplumber_failed", path=str(path), error=str(exc))
            return None

    def _parse_ocr(self, path: Path) -> ParsedPDF:
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as exc:
            raise OCRFailureError(
                "OCR dependencies not installed (pdf2image / pytesseract).",
                details={"cause": str(exc)},
            ) from exc

        try:
            images = convert_from_path(str(path), dpi=self._settings.ocr_dpi)
        except Exception as exc:  # noqa: BLE001
            raise OCRFailureError(
                f"pdf2image failed to rasterise {path}",
                details={"cause": str(exc)},
            ) from exc

        if not images:
            raise OCRFailureError(f"No pages rendered from {path}")

        try:
            page_texts = [pytesseract.image_to_string(img) for img in images]
        except Exception as exc:  # noqa: BLE001
            raise OCRFailureError(
                f"pytesseract failed on {path}",
                details={"cause": str(exc)},
            ) from exc

        text = "\n".join(page_texts).strip()
        if not text:
            raise OCRFailureError(f"OCR produced no text for {path}")

        return ParsedPDF(text=text, page_count=len(images), used_ocr=True)
