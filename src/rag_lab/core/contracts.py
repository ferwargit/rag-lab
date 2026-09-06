from typing import Protocol

from rag_lab.core.models import EmbeddedChunk, SearchResult


class EmbeddingProvider(Protocol):
    """Contrato para generar embeddings."""

    def embed(
        self,
        text: str,
    ) -> list[float]:
        ...


class VectorStore(Protocol):
    """Contrato para un almacenamiento de chunks vectorizados."""

    @property
    def items(self) -> tuple[EmbeddedChunk, ...]:
        ...


class RetrieverProvider(Protocol):
    """Contrato para recuperar documentos."""

    def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        ...