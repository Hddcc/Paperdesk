"""Text extraction for safe, session-scoped file assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from app.domains.paper.pdf_parser import ParsedPdfDocument, PdfParser


@dataclass(slots=True)
class TextExtractionResult:
    status: str
    text: str = ""
    failure_reason: str | None = None


class FileTextExtractor:
    """Extract plain text without executing uploaded content."""

    TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "cp1252")
    LOW_QUALITY_PDF_REASON = (
        "PDF text extraction produced too little usable text. "
        "This file may be scanned or image-based; OCR is not enabled."
    )
    _PAGE_NUMBER_ONLY_RE = re.compile(
        r"^\s*(?:[-–—_]*\s*)?(?:第\s*)?\d{1,4}(?:\s*页)?"
        r"(?:\s*(?:/|of)\s*\d{1,4})?(?:\s*[-–—_]*)?\s*$",
        re.IGNORECASE,
    )

    def __init__(self, pdf_parser: PdfParser | None = None) -> None:
        self.pdf_parser = pdf_parser or PdfParser()

    def extract(self, file_path: Path, *, kind: str) -> TextExtractionResult:
        if kind in {"txt", "md"}:
            return self._extract_plain_text(file_path)
        if kind == "docx":
            return self._extract_docx(file_path)
        if kind == "pdf":
            return self._extract_pdf(file_path)
        return TextExtractionResult(status="skipped", failure_reason="Unsupported file type")

    def _extract_plain_text(self, file_path: Path) -> TextExtractionResult:
        data = file_path.read_bytes()
        for encoding in self.TEXT_ENCODINGS:
            try:
                return TextExtractionResult(
                    status="ready",
                    text=self._normalize_text(data.decode(encoding)),
                )
            except UnicodeDecodeError:
                continue
        return TextExtractionResult(
            status="failed",
            failure_reason="Unable to decode text file with supported encodings",
        )

    def _extract_docx(self, file_path: Path) -> TextExtractionResult:
        try:
            from docx import Document
        except ImportError:
            return TextExtractionResult(
                status="skipped",
                failure_reason="DOCX text extraction dependency is not installed",
            )

        try:
            document = Document(str(file_path))
            paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        except Exception as exc:
            return TextExtractionResult(
                status="failed",
                failure_reason=f"DOCX text extraction failed: {exc}",
            )

        text = self._normalize_text("\n\n".join(paragraphs))
        if not text:
            return TextExtractionResult(status="failed", failure_reason="DOCX contained no extractable paragraph text")
        return TextExtractionResult(status="ready", text=text)

    def _extract_pdf(self, file_path: Path) -> TextExtractionResult:
        try:
            parsed = self.pdf_parser.parse(file_path)
        except Exception as exc:
            return TextExtractionResult(
                status="failed",
                failure_reason=f"PDF text extraction failed: {exc}",
            )

        page_blocks = [
            f"[Page {page.page_number}]\n{page.text}"
            for page in parsed.pages
            if page.text.strip()
        ]
        text = self._normalize_text("\n\n".join(page_blocks))
        if not text:
            return TextExtractionResult(status="failed", failure_reason="PDF contained no extractable text")
        if self._has_low_pdf_text_quality(parsed):
            return TextExtractionResult(
                status="failed",
                text=text,
                failure_reason=self.LOW_QUALITY_PDF_REASON,
            )
        return TextExtractionResult(status="ready", text=text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\u00a0", " ")
        lines = [line.rstrip() for line in cleaned.splitlines()]
        return "\n".join(lines).strip()

    @classmethod
    def _has_low_pdf_text_quality(cls, parsed: ParsedPdfDocument) -> bool:
        if parsed.page_count <= 0:
            return False

        raw_text = "\n".join(page.text for page in parsed.pages if page.text.strip())
        text_char_count = len(raw_text)
        extracted_page_count = len(parsed.pages)
        avg_chars_per_page = text_char_count / parsed.page_count
        image_page_ratio = parsed.image_page_count / parsed.page_count
        page_number_like_page_count = sum(
            1
            for page in parsed.pages
            if cls._is_page_number_like_text(page.text)
        )
        page_number_like_text_ratio = (
            page_number_like_page_count / extracted_page_count
            if extracted_page_count
            else 0.0
        )
        non_page_number_text_chars = cls._non_page_number_text_chars(parsed)

        if parsed.page_count >= 2 and text_char_count < 200:
            return True
        if parsed.page_count >= 2 and avg_chars_per_page < 50 and image_page_ratio >= 0.5:
            return True
        if extracted_page_count >= 2 and page_number_like_text_ratio >= 0.7:
            return True
        if parsed.page_count >= 2 and image_page_ratio >= 0.8 and non_page_number_text_chars < 100:
            return True
        if image_page_ratio >= 0.8 and non_page_number_text_chars < 20:
            return True
        return False

    @classmethod
    def _is_page_number_like_text(cls, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return bool(lines) and all(cls._PAGE_NUMBER_ONLY_RE.match(line) for line in lines)

    @classmethod
    def _non_page_number_text_chars(cls, parsed: ParsedPdfDocument) -> int:
        count = 0
        for page in parsed.pages:
            for line in page.text.splitlines():
                stripped = line.strip()
                if not stripped or cls._PAGE_NUMBER_ONLY_RE.match(stripped):
                    continue
                count += len(re.sub(r"\s+", "", stripped))
        return count
