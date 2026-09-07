"""Estrategias de generación."""

from rag_lab.generation.strategies.single_call import (
    SINGLE_CALL_SYSTEM_INSTRUCTION,
    SingleCallRAG,
    SingleCallResult,
)

__all__ = [
    "SINGLE_CALL_SYSTEM_INSTRUCTION",
    "SingleCallRAG",
    "SingleCallResult",
]