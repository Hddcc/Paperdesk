"""Chunk parsed PDF text into retrieval-friendly segments."""

from __future__ import annotations

import re

from app.models import ChunkRecord, LibraryDocument

from .pdf_parser import ParsedPdfPage

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_SENTENCE_BREAK = re.compile(r"(?<=[。！？.!?；;])\s+")


class TextChunker:
    """Split page text into recursive, overlap-aware chunks."""

    def __init__(
        self,
        *,
        target_size: int = 1000,
        overlap: int = 120,
        min_chunk_size: int = 120,
    ) -> None:
        self.target_size = target_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(
        self,
        *,
        document: LibraryDocument,
        pages: list[ParsedPdfPage],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        for page in pages:
            page_chunks = self._split_page(page.text)
            for text in page_chunks:
                chunk_id = f"{document.id}-v{document.version}-chunk-{chunk_index:04d}"
                content = text.strip()
                chunks.append(
                    ChunkRecord(
                        id=chunk_id,
                        chunk_id=chunk_id,
                        document_id=document.id,
                        source=document.file_path,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        title=document.title or document.display_name or document.filename,
                        sha256=document.sha256,
                        version=document.version,
                        text=content,
                        content=content,
                        token_estimate=max(len(content) // 4, 1),
                        metadata={
                            "document_id": document.id,
                            "filename": document.display_name or document.filename,
                            "page_number": page.page_number,
                            "chunk_index": chunk_index,
                            "title": document.title or document.display_name or document.filename,
                            "file_path": document.file_path,
                            "source": document.file_path,
                            "source_type": "local_document",
                            "sha256": document.sha256,
                            "version": document.version,
                        },
                    )
                )
                chunk_index += 1

        return chunks

    def _split_page(self, text: str) -> list[str]:
        normalized = text.strip()
        if len(normalized) <= self.target_size:
            return [normalized] if normalized else []

        chunks = self._split_segments(
            segments=self._split_paragraphs(normalized),
            splitter=self._split_sentences,
        )
        return self._merge_small_chunks(chunks)

    def _split_segments(
        self,
        *,
        segments: list[str],
        splitter,
    ) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for segment in segments:
            cleaned = segment.strip()
            if not cleaned:
                continue
            if len(cleaned) > self.target_size:
                oversized = splitter(cleaned)
                if oversized == [cleaned]:
                    chunks.extend(self._sliding_window(cleaned))
                else:
                    chunks.extend(
                        self._split_segments(segments=oversized, splitter=self._sliding_window_split)
                    )
                continue

            separator_length = 2 if current else 0
            projected = current_length + separator_length + len(cleaned)
            if projected <= self.target_size:
                current.append(cleaned)
                current_length = projected
                continue

            if current:
                chunks.append("\n\n".join(current).strip())
                current = self._overlap_seed(current)
                current_length = len("\n\n".join(current)) if current else 0

            if cleaned:
                if len(cleaned) >= self.target_size:
                    chunks.extend(self._sliding_window(cleaned))
                    current = []
                    current_length = 0
                else:
                    current.append(cleaned)
                    current_length = len(cleaned)

        if current:
            chunks.append("\n\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _split_paragraphs(self, text: str) -> list[str]:
        parts = [part.strip() for part in _PARAGRAPH_BREAK.split(text) if part.strip()]
        return parts or [text]

    def _split_sentences(self, text: str) -> list[str]:
        parts = [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]
        return parts or [text]

    def _sliding_window_split(self, text: str) -> list[str]:
        if len(text) <= self.target_size:
            return [text]
        return self._sliding_window(text)

    def _sliding_window(self, text: str) -> list[str]:
        chunks: list[str] = []
        cursor = 0
        text_length = len(text)
        while cursor < text_length:
            end = min(cursor + self.target_size, text_length)
            chunk = text[cursor:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            cursor = max(end - self.overlap, cursor + 1)
        return chunks or [text]

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []

        merged: list[str] = []
        buffer = chunks[0]
        for chunk in chunks[1:]:
            if len(buffer) < self.min_chunk_size:
                candidate = f"{buffer}\n\n{chunk}".strip()
                if len(candidate) <= self.target_size + self.overlap:
                    buffer = candidate
                    continue
            merged.append(buffer)
            buffer = chunk
        merged.append(buffer)
        return merged

    def _overlap_seed(self, segments: list[str]) -> list[str]:
        if self.overlap <= 0 or not segments:
            return []

        seed: list[str] = []
        total = 0
        for segment in reversed(segments):
            length = len(segment) + (2 if seed else 0)
            if total + length > self.overlap and seed:
                break
            seed.insert(0, segment)
            total += length
            if total >= self.overlap:
                break
        return seed
