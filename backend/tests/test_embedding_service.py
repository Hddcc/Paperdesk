from app.services.embedding_service import EmbeddingService


class FakeArray:
    def __init__(self, values):
        self._values = values

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def tolist(self):
        return self._values


def test_normalize_output_accepts_dense_vec_arrays_without_truthiness_checks():
    output = {
        "dense_vecs": FakeArray([[1, 2, 3], [4.5, 5.5, 6.5]]),
    }

    vectors = EmbeddingService._normalize_output(output)

    assert vectors == [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]]


def test_embedding_load_error_includes_cache_and_mirror_guidance(tmp_path):
    service = EmbeddingService(
        "BAAI/bge-m3",
        cache_dir=tmp_path / "hf-cache",
        hf_endpoint="https://hf-mirror.example/",
        local_files_only=True,
    )

    message = service._format_load_error(TimeoutError("timed out"))

    assert "Failed to load embedding model 'BAAI/bge-m3'" in message
    assert "EMBEDDING_HF_ENDPOINT=https://hf-mirror.com" in message
    assert "EMBEDDING_LOCAL_FILES_ONLY=true" in message
    assert "https://hf-mirror.example" in message
    assert str(tmp_path / "hf-cache") in message


def test_embedding_service_prefers_default_huggingface_cache(monkeypatch, tmp_path):
    hub_dir = tmp_path / "hub"
    snapshot_dir = hub_dir / "models--BAAI--bge-m3" / "snapshots" / "local-snapshot"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_dir))

    service = EmbeddingService("BAAI/bge-m3")

    assert service._has_cached_model() is True


def test_embedding_service_does_not_force_local_only_for_custom_cache(tmp_path):
    snapshot_dir = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "local-snapshot"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    service = EmbeddingService("BAAI/bge-m3", cache_dir=tmp_path)

    assert service._has_cached_model() is False
