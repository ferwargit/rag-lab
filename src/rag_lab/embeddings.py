"""Compatibilidad temporal: los embeddings viven en infraestructura."""

from rag_lab.infrastructure.embeddings.lm_studio import (
    EmbeddingError,
    LocalEmbeddingClient,
)

__all__ = [
    "EmbeddingError",
    "LocalEmbeddingClient",
]
