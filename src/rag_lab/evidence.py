from dataclasses import dataclass

from rag_lab.retrieval import SearchResult


@dataclass(frozen=True)
class EvidenceAssessment:
    """Representa una evaluación preliminar de la evidencia recuperada."""

    results: tuple[SearchResult, ...]
    has_candidates: bool
    best_score: float | None

def assess_evidence(
    results: list[SearchResult],
) -> EvidenceAssessment:
    """Evalúa de forma preliminar los resultados recuperados."""

    ordered_results = tuple(results)

    if not ordered_results:
        return EvidenceAssessment(
            results=ordered_results,
            has_candidates=False,
            best_score=None,
        )

    return EvidenceAssessment(
        results=ordered_results,
        has_candidates=True,
        best_score=ordered_results[0].score,
    )