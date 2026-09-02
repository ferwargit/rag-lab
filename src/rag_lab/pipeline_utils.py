from collections.abc import Sequence

from rag_lab.retrieval import SearchResult


def select_results(
    results: Sequence[SearchResult],
    selected_chunk_ids: Sequence[str],
) -> list[SearchResult]:
    """Selecciona resultados recuperados por sus IDs.

    Conserva el orden original de `results`.
    """
    selected_ids = set(selected_chunk_ids)

    return [
        result
        for result in results
        if result.chunk.id in selected_ids
    ]