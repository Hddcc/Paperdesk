"""Lazy local embedding generation for PDF retrieval."""

from __future__ import annotations

from threading import Lock

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate normalized embeddings with a lazily loaded local model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = Lock()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            raise RuntimeError(
                f"Failed to generate document embeddings with '{self.model_name}': {exc}"
            ) from exc
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        model = self._get_model()
        try:
            vector = model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            raise RuntimeError(
                f"Failed to generate query embedding with '{self.model_name}': {exc}"
            ) from exc
        return vector.tolist()

    def _get_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as exc:  # pragma: no cover - depends on local model availability
                    raise RuntimeError(
                        f"Failed to load embedding model '{self.model_name}': {exc}"
                    ) from exc
        return self._model
