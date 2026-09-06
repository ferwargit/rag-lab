"""Compatibilidad temporal: el chunking vive en rag_lab.ingestion.chunking."""

from rag_lab.ingestion.chunking import chunk_text

__all__ = ["chunk_text"]