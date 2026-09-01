import json
from pathlib import Path

import pytest

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


INDEX_PATH = Path("storage/index.json")
BENCHMARK_PATH = Path("data/benchmark.json")


def load_benchmark() -> list[dict]:
    """Carga los casos de evaluación desde benchmark.json."""

    if not BENCHMARK_PATH.exists():
        pytest.fail(
            f"No existe el benchmark: {BENCHMARK_PATH}.",
        )

    try:
        data = json.loads(
            BENCHMARK_PATH.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"El benchmark no contiene JSON válido: {exc}"
        )

    if not isinstance(data, list):
        pytest.fail(
            "El benchmark debe contener una lista de casos."
        )

    return data


@pytest.fixture
def retriever() -> Retriever:
    """Carga el índice vectorial para los tests."""

    if not INDEX_PATH.exists():
        pytest.fail(
            f"No existe el índice: {INDEX_PATH}. "
            "Ejecuta primero la indexación."
        )

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    return Retriever(store)


@pytest.fixture
def embedding_client() -> LocalEmbeddingClient:
    """Crea el cliente de embeddings local."""

    return LocalEmbeddingClient()


BENCHMARK_CASES = [
    case
    for case in load_benchmark()
    if case.get("answerable") is True
]


@pytest.mark.parametrize(
    "case",
    BENCHMARK_CASES,
    ids=lambda case: case["id"],
)
def test_answerable_queries_retrieve_expected_chunk(
    case: dict,
    retriever: Retriever,
    embedding_client: LocalEmbeddingClient,
) -> None:
    """Verifica que las consultas respondibles recuperen evidencia esperada."""

    query = case["query"]

    expected_ids = {
        chunk_id
        for chunk_id in case["expected_chunk_ids"]
    }

    embedding = embedding_client.embed(query)

    results = retriever.search(
        tuple(embedding),
        top_k=3,
        score_threshold=None,
    )

    retrieved_ids = {
        result.chunk.id
        for result in results
    }

    assert expected_ids.intersection(
        retrieved_ids
    ), (
        f"No se recuperó ningún chunk esperado para "
        f"{case['id']}: {query!r}. "
        f"Esperados={sorted(expected_ids)}, "
        f"recuperados={sorted(retrieved_ids)}"
    )