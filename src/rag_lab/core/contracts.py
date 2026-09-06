from typing import Protocol

from rag_lab.core.models import EmbeddedChunk


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