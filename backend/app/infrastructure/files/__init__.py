"""File-system adapter boundary."""

from .asset import FileAssetService
from .context_store import ContextFileStore
from .text_extractor import FileTextExtractor, TextExtractionResult

__all__ = ["ContextFileStore", "FileAssetService", "FileTextExtractor", "TextExtractionResult"]
