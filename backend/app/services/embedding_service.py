"""Lazy local embedding generation for PDF retrieval."""

from __future__ import annotations

from threading import Lock


class EmbeddingService:
    """Generate normalized embeddings with a lazily loaded FlagEmbedding model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = Lock()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            output = model.encode(texts, batch_size=12, max_length=2048)
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            raise RuntimeError(
                f"Failed to generate document embeddings with '{self.model_name}': {exc}"
            ) from exc
        return self._normalize_output(output)

    def embed_query(self, query: str) -> list[float]:
        model = self._get_model()
        try:
            output = model.encode([query], batch_size=1, max_length=1024)
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            raise RuntimeError(
                f"Failed to generate query embedding with '{self.model_name}': {exc}"
            ) from exc
        vectors = self._normalize_output(output)
        return vectors[0]

    def preload(self) -> None:
        """Download and initialize the embedding model ahead of the first upload/query."""
        self._get_model()

    def _get_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                try:
                    from FlagEmbedding import BGEM3FlagModel
                except Exception as exc:  # pragma: no cover - depends on installed packages
                    raise RuntimeError(
                        "FlagEmbedding is required for PaperDesk 09.5 embeddings. "
                        "Install it with the project dependencies before indexing documents."
                    ) from exc

                try:
                    self._model = BGEM3FlagModel(
                        self.model_name,
                        use_fp16=False,
                    )
                except Exception as exc:  # pragma: no cover - depends on local model availability
                    raise RuntimeError(
                        f"Failed to load embedding model '{self.model_name}': {exc}"
                    ) from exc
        return self._model

    @staticmethod
    def _normalize_output(output) -> list[list[float]]:
        dense_vectors = output
        if isinstance(output, dict):
            dense_vectors = []
            for key in ("dense_vecs", "dense_embeddings", "embeddings"):
                candidate = output.get(key)
                if candidate is not None:
                    dense_vectors = candidate
                    break
        if hasattr(dense_vectors, "tolist"):
            dense_vectors = dense_vectors.tolist()
        return [list(map(float, vector)) for vector in dense_vectors]
