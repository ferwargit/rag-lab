from dataclasses import dataclass

from rag_lab.models import EmbeddedChunk
from rag_lab.vector_store import JsonVectorStore


@dataclass(frozen=True)
class SearchResult:
    """Resultado de una búsqueda semántica."""

    chunk: EmbeddedChunk
    score: float


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

    def __init__(self, store: JsonVectorStore) -> None:
        self.store = store

    def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Devuelve los chunks más similares."""

        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que 0.")

        results: list[SearchResult] = []

        for chunk in self.store.items:
            score = cosine_similarity(
                query_embedding,
                chunk.embedding,
            )

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