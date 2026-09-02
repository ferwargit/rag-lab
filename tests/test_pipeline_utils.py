from rag_lab.models import EmbeddedChunk
from rag_lab.pipeline_utils import select_results
from rag_lab.retrieval import SearchResult


def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    chunk = EmbeddedChunk(
        id=chunk_id,
        text=f"Texto de {chunk_id}.",
        source="test.txt",
        index=0,
        embedding=(1.0, 0.0, 0.0),
        metadata={},
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_select_results_returns_selected_chunks() -> None:
    results = [
        make_result("chunk-001", 0.9),
        make_result("chunk-002", 0.8),
        make_result("chunk-003", 0.7),
    ]

    selected = select_results(
        results,
        ("chunk-001", "chunk-003"),
    )

    assert [result.chunk.id for result in selected] == [
        "chunk-001",
        "chunk-003",
    ]


def test_select_results_preserves_retrieval_order() -> None:
    results = [
        make_result("chunk-002", 0.9),
        make_result("chunk-001", 0.8),
        make_result("chunk-003", 0.7),
    ]

    selected = select_results(
        results,
        ("chunk-001", "chunk-002"),
    )

    assert [result.chunk.id for result in selected] == [
        "chunk-002",
        "chunk-001",
    ]


def test_select_results_with_no_selected_ids_returns_empty() -> None:
    results = [
        make_result("chunk-001", 0.9),
        make_result("chunk-002", 0.8),
    ]

    selected = select_results(
        results,
        (),
    )

    assert selected == []


def test_select_results_does_not_modify_input() -> None:
    results = [
        make_result("chunk-001", 0.9),
        make_result("chunk-002", 0.8),
    ]

    original = list(results)

    selected = select_results(
        results,
        ("chunk-002",),
    )

    assert results == original
    assert selected is not results