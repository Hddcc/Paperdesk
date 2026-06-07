"""Paper domain facade over existing services."""

from __future__ import annotations

from app.infrastructure.llm import EmbeddingService

from .ingestion import KnowledgeIngestionService
from .library import DocumentLibraryService
from .pdf_parser import PdfParser
from .rag import RagService
from .reports import ReportLifecycleService
from .text_chunker import TextChunker


class PaperDomainFacade:
    """Named paper business boundary for Agent capabilities and API use cases."""

    upload_service: type[DocumentLibraryService] = DocumentLibraryService
    parser: type[PdfParser] = PdfParser
    chunker: type[TextChunker] = TextChunker
    embedding_service: type[EmbeddingService] = EmbeddingService
    ingestion_service: type[KnowledgeIngestionService] = KnowledgeIngestionService
    rag_service: type[RagService] = RagService
    report_service: type[ReportLifecycleService] = ReportLifecycleService
