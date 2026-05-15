"""Vectorstore abstractions for local evidence retrieval."""

from .base import AbstractVectorStore
from .milvus_store import MilvusVectorStore

__all__ = ["AbstractVectorStore", "MilvusVectorStore"]
