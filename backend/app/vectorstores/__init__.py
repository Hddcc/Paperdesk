"""Vectorstore abstractions for local evidence retrieval."""

from .base import AbstractVectorStore
from .stub_vectorstore import StubVectorStore

__all__ = ["AbstractVectorStore", "StubVectorStore"]

