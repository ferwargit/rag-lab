from rag_lab.core.contracts import VectorStore
from rag_lab.core.models import SearchResult


def cosine_similarity(
    a: tuple[float, ...],
    b: tuple[float, ...],
) -> float:
    """Calcula la similitud coseno entre dos vectores."""

    if len(a) != len(b):
        raise ValueError(
            "Los vectores deben tener la misma dimensionalidad."
        )

    dot = sum(
        x * y
        for x, y in zip(a, b, strict=True)
    )

    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "No se puede calcular similitud con un vector de norma cero."
        )

    return dot / (norm_a * norm_b)


class Retriever:
    """Realiza búsqueda semántica sobre un VectorStore."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Devuelve los chunks más similares.

        Si score_threshold está definido, se descartan
        resultados cuyo score sea inferior al mínimo.
        """

        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que 0.")

        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError(
                "score_threshold debe estar entre 0 y 1."
            )

        results: list[SearchResult] = []

        for chunk in self.store.items:
            score = cosine_similarity(
                query_embedding,
                chunk.embedding,
            )

            if (
                score_threshold is not None
                and score < score_threshold
            ):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]