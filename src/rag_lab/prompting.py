"""Compatibilidad temporal: el prompting vive en rag_lab.generation.prompting."""

from rag_lab.generation.prompting import (
    SYSTEM_INSTRUCTION,
    build_rag_messages,
)

__all__ = [
    "SYSTEM_INSTRUCTION",
    "build_rag_messages",
]