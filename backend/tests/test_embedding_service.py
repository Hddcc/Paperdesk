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
