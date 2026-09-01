from rag_lab.evidence import assess_evidence
from rag_lab.models import EmbeddedChunk
from rag_lab.retrieval import SearchResult


def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    chunk = EmbeddedChunk(
        id=chunk_id,
        text="Texto de prueba.",
        source="test.txt",
        index=0,
        embedding=(1.0, 0.0, 0.0),
        metadata={},
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_assess_evidence_with_results() -> None:
    results = [
        make_result("chunk-001", 0.8),
        make_result("chunk-002", 0.7),
    ]

    assessment = assess_evidence(results)

    assert assessment.has_candidates is True
    assert assessment.best_score == 0.8
    assert len(assessment.results) == 2


def test_assess_evidence_without_results() -> None:
    assessment = assess_evidence([])

    assert assessment.has_candidates is False
    assert assessment.best_score is None
    assert assessment.results == ()