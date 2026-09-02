from pathlib import Path

import pytest

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.evidence_evaluator import EvidenceEvaluator
from rag_lab.generation import LocalChatClient
from rag_lab.rag_pipeline import ABSTENTION_MESSAGE, RAGPipeline
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


INDEX_PATH = Path("storage/index.json")


@pytest.mark.integration
def test_rag_pipeline_end_to_end_answerable_query() -> None:
    embedding_client = LocalEmbeddingClient()

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    retriever = Retriever(store)

    evidence_evaluator = EvidenceEvaluator(
        LocalChatClient()
    )

    chat_client = LocalChatClient()

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
        top_k=3,
    )

    result = pipeline.ask(
        "¿Cómo se conecta el piano al ordenador?"
    )

    assert result.sufficient is True

    assert "knowledge-001" in result.retrieved_chunk_ids
    assert "knowledge-001" in result.selected_chunk_ids

    assert result.answer.strip()
    assert result.answer != ABSTENTION_MESSAGE


@pytest.mark.integration
def test_rag_pipeline_end_to_end_abstains_when_evidence_is_missing() -> None:
    embedding_client = LocalEmbeddingClient()

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    retriever = Retriever(store)

    evidence_evaluator = EvidenceEvaluator(
        LocalChatClient()
    )

    chat_client = LocalChatClient()

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
        top_k=3,
    )

    result = pipeline.ask(
        "¿Qué sistema operativo utiliza MIDI Laboratory?"
    )

    assert result.sufficient is False
    assert result.answer == ABSTENTION_MESSAGE

    assert result.retrieved_chunk_ids
    assert result.selected_chunk_ids == ()