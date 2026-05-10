"""Parse local PDF files into page-level text blocks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import fitz


@dataclass(slots=True)
class ParsedPdfPage:
    """Normalized text content for a single PDF page."""

    page_number: int
    text: str


@dataclass(slots=True)
class ParsedPdfDocument:
    """Structured text extracted from a PDF file."""

    title: str | None
    page_count: int
    pages: list[ParsedPdfPage]


class PdfParser:
    """Extract page text from PDFs with lightweight normalization."""

    def parse(self, file_path: Path) -> ParsedPdfDocument:
        try:
            document = fitz.open(file_path)
        except Exception as exc:  # pragma: no cover - passthrough for runtime issues
            raise RuntimeError(f"Unable to open PDF '{file_path.name}': {exc}") from exc

        try:
            title = self._clean_title(document.metadata.get("title"))
            pages: list[ParsedPdfPage] = []
            for page_index, page in enumerate(document, start=1):
                text = self._normalize_page_text(page.get_text("text"))
                if not text:
                    continue
                pages.append(ParsedPdfPage(page_number=page_index, text=text))

            return ParsedPdfDocument(
                title=title,
                page_count=document.page_count,
                pages=pages,
            )
        finally:
            document.close()

    @staticmethod
    def _clean_title(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _normalize_page_text(text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\u00a0", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        lines = [line.strip() for line in cleaned.splitlines()]
        compact = "\n".join(line for line in lines if line)
        compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
        return compact
