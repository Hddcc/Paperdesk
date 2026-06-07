"""Paper domain pack.

This package is the stable boundary for paper upload, parsing, chunking,
embedding, retrieval, evidence assembly, library metadata, tags/categories, and
report-related paper behavior. Implementations are kept behind facades while
legacy service files are migrated gradually.
"""

from .analysis import PaperAnalysisService
from .facade import PaperDomainFacade
from .ingestion import KnowledgeIngestionService
from .library import DocumentLibraryService
from .pdf_parser import ParsedPdfDocument, ParsedPdfPage, PdfParser
from .query_translation import QueryTranslationService
from .rag import RagService, RetrievalResult
from .reports import ReportLifecycleService
from .search import PaperSearchService
from .selection import PaperSelectionService
from .text_chunker import TextChunker

__all__ = [
    "DocumentLibraryService",
    "KnowledgeIngestionService",
    "PaperAnalysisService",
    "PaperDomainFacade",
    "PaperSearchService",
    "PaperSelectionService",
    "ParsedPdfDocument",
    "ParsedPdfPage",
    "PdfParser",
    "QueryTranslationService",
    "RagService",
    "ReportLifecycleService",
    "RetrievalResult",
    "TextChunker",
]
