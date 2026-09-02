from pathlib import Path

import pytest

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.evidence_evaluator import EvidenceEvaluator
from rag_lab.generation import LocalChatClient
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


INDEX_PATH = Path("storage/index.json")


@pytest.fixture
def retriever() -> Retriever:
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
    return LocalEmbeddingClient()


@pytest.fixture
def evidence_evaluator() -> EvidenceEvaluator:
    return EvidenceEvaluator(
        LocalChatClient()
    )


@pytest.mark.integration
def test_evaluator_selects_required_chunks(
    retriever: Retriever,
    embedding_client: LocalEmbeddingClient,
    evidence_evaluator: EvidenceEvaluator,
) -> None:
    query = "¿Cómo se conecta el piano al ordenador?"

    query_embedding = embedding_client.embed(query)

    results = retriever.search(
        tuple(query_embedding),
        top_k=3,
        score_threshold=None,
    )

    decision = evidence_evaluator.evaluate(
        query,
        results,
    )

    assert decision.sufficient is True

    assert "knowledge-001" in (
        decision.selected_chunk_ids
    )