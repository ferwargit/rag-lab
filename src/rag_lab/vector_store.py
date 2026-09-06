"""Compatibilidad temporal: el vector store vive en infraestructura."""

from rag_lab.infrastructure.vector_store.json import (
    JsonVectorStore,
    VectorStoreError,
)

__all__ = [
    "JsonVectorStore",
    "VectorStoreError",
]