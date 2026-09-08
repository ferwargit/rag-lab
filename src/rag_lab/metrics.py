"""Compatibilidad temporal: las métricas viven en rag_lab.evaluation.metrics."""

from rag_lab.evaluation.metrics import (
    ComponentExecutionMetrics,
    ExecutionMetrics,
    metrics_from_generation,
)

__all__ = [
    "ComponentExecutionMetrics",
    "ExecutionMetrics",
    "metrics_from_generation",
]