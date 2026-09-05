"""Compatibilidad temporal: los modelos viven en rag_lab.core.models."""

from rag_lab.core.models import DocumentChunk, EmbeddedChunk, RAGResult

__all__ = ["DocumentChunk", "EmbeddedChunk", "RAGResult"]
