"""Compatibilidad temporal: los contratos viven en rag_lab.core.contracts."""

from rag_lab.core.contracts import (
    ChatGenerator,
    EmbeddingProvider,
    EvidenceEvaluatorProvider,
    RetrieverProvider,
)

__all__ = [
    "ChatGenerator",
    "EmbeddingProvider",
    "EvidenceEvaluatorProvider",
    "RetrieverProvider",
]