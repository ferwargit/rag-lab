"""Compatibilidad temporal: el pipeline vive en rag_lab.pipeline.rag_pipeline."""

from rag_lab.pipeline.rag_pipeline import (
    ABSTENTION_MESSAGE,
    RAGPipeline,
)

__all__ = [
    "ABSTENTION_MESSAGE",
    "RAGPipeline",
]