import pytest

from rag_lab.retrieval import cosine_similarity


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    vector = (1.0, 2.0, 3.0)

    score = cosine_similarity(vector, vector)

    assert score == pytest.approx(1.0)


def test_cosine_similarity_rejects_different_dimensions() -> None:
    vector_a = (1.0, 2.0, 3.0)
    vector_b = (1.0, 2.0)

    with pytest.raises(ValueError, match="misma dimensionalidad"):
        cosine_similarity(vector_a, vector_b)


def test_cosine_similarity_rejects_zero_vector() -> None:
    vector_a = (0.0, 0.0, 0.0)
    vector_b = (1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="norma cero"):
        cosine_similarity(vector_a, vector_b)