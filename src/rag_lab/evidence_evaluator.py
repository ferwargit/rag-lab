"""Compatibilidad temporal: el evaluador de evidencia vive en rag_lab.evaluation."""

from rag_lab.core.models import EvidenceDecision
from rag_lab.evaluation.evidence_evaluator import (
    EvidenceEvaluator,
)

__all__ = [
    "EvidenceDecision",
    "EvidenceEvaluator",
]