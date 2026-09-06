"""Compatibilidad temporal: el indexing vive en rag_lab.ingestion.indexing."""

from rag_lab.ingestion.indexing import embed_chunk, embed_chunks

__all__ = [
    "embed_chunk",
    "embed_chunks",
]