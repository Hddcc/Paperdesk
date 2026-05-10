"""Chunk parsed PDF text into retrieval-friendly segments."""

from __future__ import annotations

from app.models import ChunkRecord

from .pdf_parser import ParsedPdfPage


class TextChunker:
    """Split page text into moderately sized overlapping chunks."""

    def __init__(
        self,
        *,
        target_size: int = 1000,
        overlap: int = 180,
        min_chunk_size: int = 120,
    ) -> None:
        self.target_size = target_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(
        self,
        *,
        document_id: str,
        filename: str,
        title: str,
        file_path: str,
        pages: list[ParsedPdfPage],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        for page in pages:
            page_chunks = self._split_page(page.text)
            for text in page_chunks:
                chunk_id = f"{document_id}-chunk-{chunk_index:04d}"
                chunks.append(
                    ChunkRecord(
                        id=chunk_id,
                        document_id=document_id,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=text,
                        token_estimate=max(len(text) // 4, 1),
                        metadata={
                            "document_id": document_id,
                            "filename": filename,
                            "page_number": page.page_number,
                            "chunk_index": chunk_index,
                            "title": title,
                            "file_path": file_path,
                            "source_type": "local_document",
                        },
                    )
                )
                chunk_index += 1

        return chunks

    def _split_page(self, text: str) -> list[str]:
        if len(text) <= self.target_size:
            return [text]

        chunks: list[str] = []
        cursor = 0
        text_length = len(text)

        while cursor < text_length:
            end = min(cursor + self.target_size, text_length)
            chunk = text[cursor:end].strip()
            if chunk and (len(chunk) >= self.min_chunk_size or not chunks):
                chunks.append(chunk)
            if end >= text_length:
                break
            next_cursor = max(end - self.overlap, cursor + 1)
            cursor = next_cursor

        return chunks or [text]
