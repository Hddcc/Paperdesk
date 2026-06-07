"""Lazy local embedding generation for PDF retrieval."""

from __future__ import annotations

from pathlib import Path
import os
from threading import Lock


class EmbeddingService:
    """Generate normalized embeddings with a lazily loaded FlagEmbedding model."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path | str | None = None,
        hf_endpoint: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hf_endpoint = hf_endpoint.strip().rstrip("/") if hf_endpoint else None
        self.local_files_only = local_files_only
        self._model = None
        self._lock = Lock()
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint

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
                prefer_local_cache = self.local_files_only or self._has_cached_model()
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
                        cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                        local_files_only=prefer_local_cache,
                    )
                except Exception as exc:  # pragma: no cover - depends on local model availability
                    raise RuntimeError(
                        self._format_load_error(exc)
                    ) from exc
        return self._model

    def _format_load_error(self, exc: Exception) -> str:
        hints = [
            f"Failed to load embedding model '{self.model_name}': {exc}",
            "PaperDesk still uses the real BGE-M3 embedding model; this is a model download/cache issue.",
            "Fix options:",
            "1. Set EMBEDDING_HF_ENDPOINT=https://hf-mirror.com and restart the backend.",
            "2. Or pre-download BAAI/bge-m3, set EMBEDDING_MODEL to that local directory, and set EMBEDDING_LOCAL_FILES_ONLY=true.",
        ]
        if self.cache_dir is not None:
            hints.append(f"Current embedding cache dir: {self.cache_dir}")
        if self.hf_endpoint:
            hints.append(f"Current HF endpoint: {self.hf_endpoint}")
        if self.local_files_only:
            hints.append("Current mode: local files only.")
        return " ".join(hints)

    def _has_cached_model(self) -> bool:
        if self.cache_dir is not None or self._looks_like_local_path(self.model_name):
            return False

        repo_cache_name = f"models--{self.model_name.replace('/', '--')}"
        for hub_dir in self._default_hub_cache_dirs():
            model_dir = hub_dir / repo_cache_name
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue
            if any(snapshot.is_dir() and (snapshot / "config.json").is_file() for snapshot in snapshots_dir.iterdir()):
                return True
        return False

    @staticmethod
    def _looks_like_local_path(model_name: str) -> bool:
        return (
            model_name.startswith((".", "/", "\\", "~"))
            or ":" in model_name
            or "\\" in model_name
        )

    @staticmethod
    def _default_hub_cache_dirs() -> list[Path]:
        candidates: list[Path] = []
        for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            value = os.environ.get(env_name)
            if value:
                candidates.append(Path(value).expanduser())

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidates.append(Path(hf_home).expanduser() / "hub")

        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            candidates.append(Path(xdg_cache_home).expanduser() / "huggingface" / "hub")

        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
        return candidates

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
