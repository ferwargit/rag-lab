from pathlib import Path

import pytest

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.evidence_evaluator import EvidenceEvaluator
from rag_lab.generation import LocalChatClient
from rag_lab.rag_pipeline import ABSTENTION_MESSAGE, RAGPipeline
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore
from rag_lab.benchmark import (
    evaluate_benchmark_case,
    load_benchmark,
    run_and_evaluate_benchmark,
    run_benchmark,
)


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


@pytest.mark.integration
@pytest.mark.parametrize(
    "query, expected_sufficient, expected_chunk_id",
    [
        (
            "¿Cómo se conecta el piano al ordenador?",
            True,
            "knowledge-001",
        ),
        (
            "¿Dónde se ejecuta la interfaz de usuario?",
            True,
            "knowledge-002",
        ),
        (
            "¿Cuál es el objetivo de la aplicación?",
            True,
            "knowledge-003",
        ),
        (
            "¿Qué sistema operativo utiliza MIDI Laboratory?",
            False,
            None,
        ),
    ],
)
def test_rag_pipeline_end_to_end_benchmark_queries(
    query: str,
    expected_sufficient: bool,
    expected_chunk_id: str | None,
) -> None:
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

    result = pipeline.ask(query)

    assert result.sufficient is expected_sufficient

    if expected_sufficient:
        assert expected_chunk_id is not None
        assert expected_chunk_id in result.retrieved_chunk_ids
        assert expected_chunk_id in result.selected_chunk_ids
        assert result.answer.strip()
        assert result.answer != ABSTENTION_MESSAGE
    else:
        assert result.selected_chunk_ids == ()
        assert result.answer == ABSTENTION_MESSAGE


BENCHMARK_PATH = Path("data/benchmark.json")


@pytest.mark.integration
def test_rag_pipeline_end_to_end_passes_benchmark() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

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

    for case in cases:
        result = pipeline.ask(case.query)

        assert evaluate_benchmark_case(
            case,
            result,
        ), f"Falló el caso del benchmark: {case.id}"


@pytest.mark.integration
def test_rag_pipeline_can_run_benchmark() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

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

    results = run_benchmark(
        cases,
        pipeline.ask,
    )

    assert len(results) == len(cases)

    assert [result.case_id for result in results] == [
        case.id for case in cases
    ]


@pytest.mark.integration
def test_rag_pipeline_end_to_end_evaluates_full_benchmark() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

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

    results, evaluations = run_and_evaluate_benchmark(
        cases,
        pipeline.ask,
    )

    assert len(results) == 4
    assert len(evaluations) == 4

    assert evaluations == [
        True,
        True,
        True,
        True,
    ]