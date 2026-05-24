"""Text extraction for safe, session-scoped file assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TextExtractionResult:
    status: str
    text: str = ""
    failure_reason: str | None = None


class FileTextExtractor:
    """Extract plain text without executing uploaded content."""

    TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "cp1252")

    def extract(self, file_path: Path, *, kind: str) -> TextExtractionResult:
        if kind in {"txt", "md"}:
            return self._extract_plain_text(file_path)
        if kind == "docx":
            return self._extract_docx(file_path)
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\u00a0", " ")
        lines = [line.rstrip() for line in cleaned.splitlines()]
        return "\n".join(lines).strip()
