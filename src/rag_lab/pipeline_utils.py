"""Compatibilidad temporal: las utilidades del pipeline viven en rag_lab.pipeline.utils."""

from rag_lab.pipeline.utils import select_results

__all__ = [
    "select_results",
]