import pytest

from rag_lab.rag_pipeline import (
    ABSTENTION_MESSAGE,
    RAGPipeline,
)


class DummyEmbeddingClient:
    pass


class DummyRetriever:
    pass


class DummyEvidenceEvaluator:
    pass


class DummyChatClient:
    pass


def test_rag_pipeline_stores_dependencies() -> None:
    embedding_client = DummyEmbeddingClient()
    retriever = DummyRetriever()
    evidence_evaluator = DummyEvidenceEvaluator()
    chat_client = DummyChatClient()

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
        top_k=3,
    )

    assert pipeline.embedding_client is embedding_client
    assert pipeline.retriever is retriever
    assert (
        pipeline.evidence_evaluator
        is evidence_evaluator
    )
    assert pipeline.chat_client is chat_client
    assert pipeline.top_k == 3


def test_rag_pipeline_rejects_invalid_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="top_k debe ser mayor que cero",
    ):
        RAGPipeline(
            embedding_client=DummyEmbeddingClient(),
            retriever=DummyRetriever(),
            evidence_evaluator=DummyEvidenceEvaluator(),
            chat_client=DummyChatClient(),
            top_k=0,
        )


def test_rag_pipeline_ask_is_not_implemented_yet() -> None:
    pipeline = RAGPipeline(
        embedding_client=DummyEmbeddingClient(),
        retriever=DummyRetriever(),
        evidence_evaluator=DummyEvidenceEvaluator(),
        chat_client=DummyChatClient(),
    )

    with pytest.raises(NotImplementedError):
        pipeline.ask(
            "Pregunta de prueba"
        )


def test_abstention_message_is_defined() -> None:
    assert ABSTENTION_MESSAGE.startswith(
        "No tengo información suficiente"
    )